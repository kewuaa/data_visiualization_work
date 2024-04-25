use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use pyo3::prelude::*;
use numpy::{prelude::*, PyArray2, PyReadonlyArray2};
use ndarray::prelude::*;


fn cal_pairwise_dist(x: ArrayView2<f64>) -> Array2<f64> {
    let sum_x = x.mapv(|v| v.powi(2)).sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let view = sum_x.broadcast((x.nrows(), x.nrows())).unwrap();
    let dist =
        -2.0 * x.dot(&x.t()) +
        view + view.t();
    dist
}


#[pyclass]
pub struct TSNE {
    perplexity: f64,
    early_exaggeration: bool,
}


impl TSNE {
    pub fn new(perplexity: f64, early_exaggeration: bool) -> TSNE {
        TSNE {
            perplexity,
            early_exaggeration
        }
    }

    fn _search_beta<F>(
        &self,
        cal_perp: F,
    ) -> f64
    where
        F: Fn(f64) -> f64
    {
        let mut beta = 1.0f64;
        let mut down_bound: Option<f64> = None;
        let mut up_bound: Option<f64> = None;
        for i in 0..50 {
            let perp = cal_perp(beta);
            let diff = perp - self.perplexity;
            if diff > 1e-5 {
                if let Some(bound) = up_bound {
                    down_bound = Some(beta);
                    beta = (beta + bound) / 2.;
                } else {
                    down_bound = Some(beta);
                    beta *= 2.;
                }
            } else if diff < -1e-5 {
                if let Some(bound) = down_bound {
                    up_bound = Some(beta);
                    beta = (beta + bound) / 2.;
                } else {
                    up_bound = Some(beta);
                    beta /= 2.;
                }
            } else {
                println!("iter {} and get best beta", i);
                break;
            }
        }
        beta
    }

    fn _cal_high_dim_prob(
        &self,
        dist: ArrayView2<f64>,
    ) -> Array2<f64> {
        let mut probilities: Array2<f64> = Array::zeros(dist.raw_dim());
        for i in 0..dist.nrows() {
            let cal_perp = |beta: f64| {
                    let mut probility = dist.row(i).mapv(|v| (-v * beta).exp());
                    probility[i] = 0.;
                    probility /= probility.sum();
                    probility.mapv(|v| (-v * v.log2()).max(1e-5)).sum().exp2()
                };
            let beta = self._search_beta(cal_perp);
            let mut probility = dist.row(i).mapv(|v| (-v * beta).exp());
            probility[i] = 0.;
            let sum = probility.sum();
            probilities.index_axis_mut(Axis(0), i)
                .assign(&(probility / sum));
        }
        let view = probilities.view();
        probilities = &view + &view.t();
        probilities /= probilities.nrows() as f64 * 2.;
        probilities
    }

    fn _cal_low_dim_prob(&self, dist: ArrayView2<f64>) -> Array2<f64> {
        let mut probilities = dist.mapv(|v| {
            1. / (1. + v)
        });
        probilities.diag_mut().fill(0.);
        let sum = probilities.sum();
        probilities / sum
    }

    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        max_iter_num: usize,
        learning_rate: f64,
        momentum: f64,
    ) -> Array2<f64>{
        let x_dist = cal_pairwise_dist(x.view());
        let mut p = self._cal_high_dim_prob(x_dist.view());
        if self.early_exaggeration {
            p *= 4.;
        }
        let mut y = Array::random((x.nrows(), 2), Uniform::new(0., 1.));
        let mut iy: Array2<f64> = Array::zeros((x.nrows(), 2));
        let shape = (x.nrows(), x.nrows(), 2);
        for i in 0..max_iter_num {
            if self.early_exaggeration && i == 50 as usize {
                p /= 4.;
            }
            let y_dist = cal_pairwise_dist(y.view());
            let q = self._cal_low_dim_prob(y_dist.view());
            let pq = (&p - &q).insert_axis(Axis(2));
            let inv_dist = y_dist.mapv(|v| 1. / (1. + v)).insert_axis(Axis(2));
            let y1 = y.clone().insert_axis(Axis(0));
            let y2 = y.clone().insert_axis(Axis(1));
            let y_diffs = &y1.broadcast(shape).unwrap() - &y2.broadcast(shape).unwrap();
            let grad = y_diffs * &pq.broadcast(shape).unwrap() * &inv_dist.broadcast(shape).unwrap() * 4.;
            iy = grad.sum_axis(Axis(1)) * learning_rate + iy * momentum;
            y += &iy;
            if i % 100 == 0 {
                let error = ((&p / q).mapv(|v| {
                    if v.is_nan() {
                        0.
                    } else {
                        v
                    }
                }) * &p).sum();
                println!("error of iter {}: {}", i, error);
            }
        }
        println!("iter done");
        y
    }
}

#[pymethods]
impl TSNE {
    #[new]
    pub fn new_py(perplexity: f64, early_exaggeration: bool) -> TSNE {
        TSNE {
            perplexity,
            early_exaggeration
        }
    }

    #[pyo3(name="fit")]
    pub fn fit_py<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        max_iter_num: usize,
        learning_rate: f64,
        momentum: f64,
    ) -> Bound<'py, PyArray2<f64>>{
        let arr = self.fit(x.as_array(), max_iter_num, learning_rate, momentum);
        arr.into_pyarray_bound(py)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cal_pairwise() {
        let x = array![
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ];
        assert_eq!(
            cal_pairwise_dist(x.view()),
            array![
                [0., 27., 108.],
                [27., 0., 27.],
                [108., 27., 0.],
            ],
        );
    }
}
