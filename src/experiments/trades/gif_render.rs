use plotters::{coord::Shift, prelude::*};

use crate::Series;

pub(crate) struct GifRender<'a> {
    root: DrawingArea<BitMapBackend<'a>, Shift>,
    fits: Vec<Series>,
    min_fit: f64,
    max_fit: f64,
    min_ts: f64,
    max_ts: f64,
}

impl<'a> GifRender<'a> {
    pub(crate) fn new(filename: &str, dims: (u32, u32), num_candidates: usize) -> Self {
        let root = BitMapBackend::gif(filename, dims, 100).unwrap().into_drawing_area();

        Self {
            root,
            fits: vec![vec![]; num_candidates],
            min_fit: 0.0,
            max_fit: 0.1,
            min_ts: 0.0,
            max_ts: 1.0,
        }
    }

    pub(crate) fn update(
        &mut self,
        targets: &Series,
        train_preds: &Series,
        test_preds: &Series,
        fits: &Vec<f64>, // current fitnesses of each candidate
        idx: usize,
        parameters: &Vec<Vec<f64>>,
    ) {
        for (i, f) in fits.iter().enumerate() {
            if *f > self.max_fit {
                self.max_fit = *f;
            }
            if *f < self.min_fit {
                self.min_fit = *f;
            }
            self.max_ts = idx as f64;
            self.fits[i].push((idx as f64, *f));
        }

        self.root.fill(&WHITE).unwrap();
        let ts_min = targets[0].0;
        let ts_max = targets[targets.len() - 1].0;
        let mut target_min: f64 = targets[0].1;
        let mut target_max: f64 = targets[targets.len() - 1].1;
        for t in targets {
            if t.1 < target_min {
                target_min = t.1;
            }
            if t.1 > target_max {
                target_max = t.1;
            }
        }
        info!("target_min: {}, target_max: {}", target_min, target_max);

        let areas = self.root.split_evenly((2, 1));
        let upper = areas[0].split_evenly((1, 2));
        let lower = areas[1].split_evenly((1, 2));

        let mut cc0 = ChartBuilder::on(&upper[0])
            .margin(5)
            .x_label_area_size(20)
            .y_label_area_size(40)
            .caption("parameters 1, 2", ("sans-serif", 20).into_font().with_color(&BLACK))
            .build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64)
            .unwrap();
        let mut cc1 = ChartBuilder::on(&upper[1])
            .margin(5)
            .x_label_area_size(20)
            .y_label_area_size(40)
            .caption("rmse", ("sans-serif", 20).into_font().with_color(&BLACK))
            .build_cartesian_2d(self.min_ts..self.max_ts, (self.min_fit..self.max_fit).log_scale())
            .unwrap();
        let mut cc2 = ChartBuilder::on(&lower[1])
            .margin(5)
            .x_label_area_size(20)
            .y_label_area_size(40)
            .caption("values", ("sans-serif", 20).into_font().with_color(&BLACK))
            .build_cartesian_2d(ts_min..ts_max, target_min..target_max)
            .unwrap();
        let mut cc3 = ChartBuilder::on(&lower[0])
            .margin(5)
            .x_label_area_size(20)
            .y_label_area_size(40)
            .caption("parameters 3, 4", ("sans-serif", 20).into_font().with_color(&BLACK))
            .build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64)
            .unwrap();

        cc0.configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .x_label_formatter(&|v| format!("{:.2}", v))
            .y_label_formatter(&|v| format!("{:.2}", v))
            .draw()
            .unwrap();
        cc1.configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .draw()
            .unwrap();
        cc2.configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.2}", v))
            .draw()
            .unwrap();
        cc3.configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .x_label_formatter(&|v| format!("{:.2}", v))
            .y_label_formatter(&|v| format!("{:.2}", v))
            .draw()
            .unwrap();

        let mut params_0_1 = Vec::with_capacity(parameters.len());
        let mut params_2_3 = Vec::with_capacity(parameters.len());
        for p in parameters {
            params_0_1.push((p[0], p[1]));
            params_2_3.push((p[2], p[3]));
        }
        cc0.draw_series(params_0_1.iter().map(|(x, y)| Circle::new((*x, *y), 2, BLACK.filled())))
            .unwrap()
            .label("params_0_1")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
        cc3.draw_series(params_2_3.iter().map(|(x, y)| Circle::new((*x, *y), 2, BLACK.filled())))
            .unwrap()
            .label("params_2_3")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

        for f in &self.fits {
            cc1.draw_series(LineSeries::new(f.clone(), &BLACK)).unwrap();
        }

        cc2.draw_series(LineSeries::new(targets.clone(), &BLACK))
            .unwrap()
            .label("targets")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
        cc2.draw_series(LineSeries::new(train_preds.clone(), &RED))
            .unwrap()
            .label("train_preds")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        cc2.draw_series(LineSeries::new(test_preds.clone(), &GREEN))
            .unwrap()
            .label("test_preds")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        cc2.configure_series_labels().border_style(&BLACK).draw().unwrap();

        self.root.present().unwrap();
    }
}
