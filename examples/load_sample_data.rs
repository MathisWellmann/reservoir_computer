use std::fs::File;

/// Load the sample data from csv file
/// output are price returns
pub(crate) fn load_sample_data() -> Vec<f64> {
    let f = File::open("data/Bitmex_XBTUSD_1M.csv").unwrap();

    let mut r = csv::Reader::from_reader(f);

    let mut out = Vec::with_capacity(1_000_000);
    for record in r.records() {
        let row = record.unwrap();

        out.push(row[1].parse::<f64>().unwrap());
    }

    out
}

fn main() {}
