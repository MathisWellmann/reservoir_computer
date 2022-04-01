/// Combines n predictions into one using averaging
pub(crate) fn ensemble_prediction(preds: &Vec<Vec<f64>>) -> Vec<f64> {
    // number of predictors to merge
    let n = preds.len() as f64;

    let mut out: Vec<f64> = preds[0].iter().map(|v| v / n).collect();

    for predicted in preds.iter().skip(1) {
        for (o, p) in out.iter_mut().zip(predicted) {
            *o += p / n;
        }
    }

    out
}
