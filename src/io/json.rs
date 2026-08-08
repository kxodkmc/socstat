//! JSON reader and writer for [`Dataset`].
//!
//! JSON interchange format — useful for web pipelines and debugging.
//!
//! ```json
//! {
//!   "variables": [
//!     {"name": "age", "data_type": "Numeric", "measure": "Scale"}
//!   ],
//!   "columns": [
//!     {"Numeric": [25.0, null, 35.0]}
//!   ]
//! }
//! ```

use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data::variable::{DataType, MeasureType, Variable};
use crate::data::{ColumnData, Dataset};
use crate::error::{SocStatError, SocStatResult};

use super::{Reader, Writer};

pub struct JsonReader;
pub struct JsonWriter;

// --- Serde models ---

#[derive(Serialize, Deserialize)]
struct JsonDataset {
    variables: Vec<JsonVariable>,
    columns: Vec<JsonColumn>,
}

#[derive(Serialize, Deserialize)]
struct JsonVariable {
    name: String,
    label: Option<String>,
    data_type: String,
    measure: String,
    width: Option<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum JsonColumn {
    #[serde(rename = "numeric")]
    Numeric(Vec<Option<f64>>),
    #[serde(rename = "text")]
    Text(Vec<Option<String>>),
}

// --- String helpers ---

fn data_type_str(dt: DataType) -> &'static str {
    match dt { DataType::Numeric => "Numeric", DataType::Text => "Text" }
}

fn measure_str(m: MeasureType) -> &'static str {
    match m { MeasureType::Nominal => "Nominal", MeasureType::Ordinal => "Ordinal", MeasureType::Scale => "Scale" }
}

fn parse_data_type(s: &str) -> SocStatResult<DataType> {
    match s {
        "Numeric" => Ok(DataType::Numeric),
        "Text" => Ok(DataType::Text),
        other => Err(SocStatError::Other(format!("unknown data_type: {other}"))),
    }
}

fn parse_measure(s: &str) -> SocStatResult<MeasureType> {
    match s {
        "Nominal" => Ok(MeasureType::Nominal),
        "Ordinal" => Ok(MeasureType::Ordinal),
        "Scale" => Ok(MeasureType::Scale),
        other => Err(SocStatError::Other(format!("unknown measure: {other}"))),
    }
}

// --- Reader/Writer impls ---

impl Reader for JsonReader {
    fn read_path(&self, path: &Path) -> SocStatResult<Dataset> {
        let file = File::open(path)?;
        let jd: JsonDataset = serde_json::from_reader(file)?;
        let mut ds = Dataset::new();

        for jv in &jd.variables {
            let var = Variable {
                name: jv.name.clone(),
                label: jv.label.clone(),
                data_type: parse_data_type(&jv.data_type)?,
                measure: parse_measure(&jv.measure)?,
                width: match jv.width { Some(w) => w, None => if parse_data_type(&jv.data_type)? == DataType::Text { 255 } else { 0 } },
                ..Default::default()
            };
            ds.add_var(var)?;
        }

        // Replace placeholder columns with actual data.
        for (i, jc) in jd.columns.into_iter().enumerate() {
            let col = match jc {
                JsonColumn::Numeric(v) => ColumnData::Numeric(v),
                JsonColumn::Text(v) => ColumnData::Text(v),
            };
            if let Some(c) = ds.column_mut(i) {
                *c = col;
            }
        }

        Ok(ds)
    }
}

impl Writer for JsonWriter {
    fn write_path(&self, ds: &Dataset, path: &Path) -> SocStatResult<()> {
        let variables: Vec<JsonVariable> = ds.variables().iter().map(|v| JsonVariable {
            name: v.name.clone(),
            label: v.label.clone(),
            data_type: data_type_str(v.data_type).to_string(),
            measure: measure_str(v.measure).to_string(),
            width: if v.width > 0 { Some(v.width) } else { None },
        }).collect();

        let columns: Vec<JsonColumn> = ds.columns().iter().map(|c| match c {
            ColumnData::Numeric(v) => JsonColumn::Numeric(v.clone()),
            ColumnData::Text(v) => JsonColumn::Text(v.clone()),
        }).collect();

        let jd = JsonDataset { variables, columns };
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &jd)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::value::Value;

    fn sample() -> Dataset {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("age").label("Age")).unwrap();
        ds.add_var(Variable::text("city")).unwrap();
        ds.push_row(vec![Value::Number(25.0), Value::Text("Beijing".into())]).unwrap();
        ds.push_row(vec![Value::Missing, Value::Text("Shanghai".into())]).unwrap();
        ds
    }

    #[test]
    fn json_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_json_roundtrip2.json");
        let ds = sample();
        JsonWriter.write_path(&ds, &path).unwrap();

        let ds2 = JsonReader.read_path(&path).unwrap();
        assert_eq!(ds2.n_vars(), 2);
        assert_eq!(ds2.n_rows(), 2);

        let ages = ds2.numeric_slice("age").unwrap();
        assert_eq!(ages[0], Some(25.0));
        assert!(ages[1].is_none());

        let cities = ds2.text_slice("city").unwrap();
        assert_eq!(cities[0].as_deref(), Some("Beijing"));
        assert_eq!(cities[1].as_deref(), Some("Shanghai"));
    }
}
