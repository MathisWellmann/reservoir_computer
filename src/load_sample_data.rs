use std::fs::File;

/// Load the sample data from csv file
pub(crate) fn load_sample_data() -> Vec<f32> {
    let f = File::open("data/Bitmex_XBTUSD_1M.csv").unwrap();

    let mut r = csv::Reader::from_reader(f);

    let mut out: Vec<f32> = Vec::with_capacity(1_000_000);
    for record in r.records() {
        let row = record.unwrap();

        out.push(row[1].parse::<f32>().unwrap());
    }

    out
}
