use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use pyo3::exceptions::{PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use raphtory::core::entities::VID;
use raphtory::prelude::{GID, GraphViewOps};
use std::sync::Arc;


use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use dyn_cc::snapshot_clustering::{SnapshotClusteringAlg, DiffGraph};

use dyn_cc::diff::build_snapshot_diffs;

const ARITY: usize = 64;


/// Holds a Python callback, wrapped as a Rust closure usable by pure-Rust code.
#[pyclass]
pub struct DynamicClustering {
    alg: dyn_cc::alg::DynamicClustering<ARITY, VID>,
    cluster_alg:
        Arc<dyn Fn(&mut SparseRowMat<usize, f64>, usize) -> (Vec<usize>, usize) + Send + Sync + 'static>,
}

#[pymethods]
impl DynamicClustering {
    #[new]
    fn new(
        callback: Bound<'_, PyAny>,
        sigma: f64,
        coreset_size: usize,
        sampling_seeds: usize,
        num_clusters: usize,
        prop_name: String,
    ) -> PyResult<Self> {
        if !callback.is_callable() {
            return Err(PyTypeError::new_err("callback must be callable"));
        }

        // Move the Python callback into a Rust `Py<PyAny>` we can use under the GIL later.
        let cb: Py<PyAny> = callback.unbind();

        // Build the Rust closure that wraps the Python callback.
        let alg = move |mat: &mut SparseRowMat<usize, f64>, clusters: usize| -> (Vec<usize>, usize) {
            #[allow(deprecated)]
            Python::with_gil(|py| {
                let symbolic = mat.symbolic();
                let nrows = symbolic.nrows();
                let ncols = symbolic.ncols();

                // Build Python lists from the CSR buffers (borrowing slices; PyList::new copies).
                let indptr_arr = PyList::new(py, symbolic.row_ptr()).unwrap();
                let indices_arr = PyList::new(py, symbolic.col_idx()).unwrap();
                let data_arr = PyList::new(py, mat.val()).unwrap();

                let sparse = py.import("scipy.sparse").expect("import scipy.sparse");
                let csr = sparse
                    .call_method1("csr_array", ((data_arr, indices_arr, indptr_arr), (nrows, ncols)))
                    .expect("build csr_array");

                let result = cb.call1(py, (csr, clusters)).expect("Python cluster callback failed");
                result.extract(py).expect("Python cluster callback returned wrong type")
            })
        };

        let arc_alg = Arc::new(alg);

        Ok(Self {
            alg: dyn_cc::alg::DynamicClustering::new(
                sigma.into(),
                coreset_size,
                sampling_seeds,
                num_clusters,
                arc_alg.clone(),
                prop_name
            ),
            cluster_alg: arc_alg,
        })
    }

    /// Placeholder: accept a Raphtory PersistentGraph and args from Python.
    
    fn cluster_subset_on_snapshots(&mut self, raphtory_graph: PersistentGraph, start: i64, end: i64, step: usize, subset: Vec<GID>) -> PyResult<(
        Vec<i64>, Vec<Vec<usize>>, Vec<usize>
    )> {

        // returns a Vec of snapshot times, a vec of snapshot labels and a vec of cluster sizes.

        let subset = subset.into_iter().map(|x| raphtory_graph.node(x).unwrap().node).collect::<Vec<_>>();

        let diffs = build_snapshot_diffs(
            &raphtory_graph,
            start,
            end,
            step,
            &self.alg.prop_name,
            1e-9
        ).unwrap();

        // we use a simple hashmap to follow the diff updates. Raphtory is too slow for this.
        let mut graph = DiffGraph::default();
        let partitions = self.alg.process_node_diffs_with_subset(&diffs, &mut graph, subset.as_slice());

        let mut times = Vec::with_capacity(partitions.len());
        let mut snapshot_labels = Vec::with_capacity(partitions.len());
        let mut cluster_sizes = Vec::with_capacity(partitions.len());

        for (t, part) in partitions{
            match part{
                dyn_cc::snapshot_clustering::PartitionOutput::All(_, _) => unreachable!(),
                dyn_cc::snapshot_clustering::PartitionOutput::Subset(labels, clusters) => {
                    times.push(t);
                    snapshot_labels.push(labels);
                    cluster_sizes.push(clusters);
                },
            }
        }

        Ok((times,snapshot_labels,cluster_sizes))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn dyn_cc_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_dummy_callback, m)?)?;
    m.add_class::<DynamicClustering>()?;
    Ok(())
}

impl DynamicClustering {
    /// Retrieve the Rust-side clustering function (for use inside Rust).
    pub fn into_alg(
        &self,
    ) -> Arc<dyn Fn(&mut SparseRowMat<usize, f64>, usize) -> (Vec<usize>, usize) + Send + Sync + 'static> {
        self.cluster_alg.clone()
    }
}

/// Build a small faer CSR matrix, invoke the stored Python callback via the Rust closure, and
/// return the result. This is a Rust-side smoke test callable from Python.
#[pyfunction]
fn test_dummy_callback(runner: &DynamicClustering) -> PyResult<(Vec<usize>, usize)> {
    // 2x4 matrix matching the Python test
    let nrows = 2usize;
    let ncols = 4usize;
    let indptr = vec![0usize, 2, 4];
    let indices = vec![0usize, 2, 1, 3];
    let data = vec![1.0f64, 2.0, 3.0, 4.0];

    let symbolic = SymbolicSparseRowMat::<usize>::new_checked(nrows, ncols, indptr, None, indices);
    let mat = SparseRowMat::new(symbolic, data);

    let alg = runner.into_alg();
    let mut mat = mat;
    Ok(alg(&mut mat, 2))
}
