use pyo3::prelude::*;

mod tsne;


#[pyfunction]
#[pyo3(name="say_hello")]
fn say_hello_from_py() {
    println!("hello world!!!");
}


#[pymodule]
#[pyo3(name="tsne")]
fn register_module<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(say_hello_from_py, m)?)?;
    m.add_class::<tsne::TSNE>()?;
    Ok(())
}
