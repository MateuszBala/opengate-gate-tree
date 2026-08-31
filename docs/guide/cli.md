# Command Line

```bash
opengate-gate-tree --help
```

## Options

| Option | Type | Required | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `--input-gate-root-file` | path | yes | file with `.root` extension | Path to the GATE ROOT input file. |
| `--output-dir` | path | yes | existing directory or new path | Directory the output file is written to. Created when missing. |
| `--output-file-title` | string | yes | non-empty string | Base name of the output file, without extension. |
| `--gate-tree` | string | yes | `Hits`, `Singles`, `Coincidences` | Tree to extract from the input file. |
| `--output-file-format` | string | yes | `root`, `hdf5`, `csv` | Format of the output file. |
| `--branches-to-extract` | list of strings | no | branch names present in the tree | Space-separated branches to keep. Every branch is kept when omitted. |

The output file is named `<output-file-title>.<tree>.<output-file-format>`, so
`--output-file-title patient_01 --gate-tree Hits --output-file-format csv`
writes `patient_01.hits.csv` into the output directory. The tree is part of the
name because one input file holds several of them.

## Examples

Extract the whole tree:

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format csv
```

Keep only the branches the analysis needs:

```bash
opengate-gate-tree \
	--input-gate-root-file ./data/simulation.root \
	--output-dir ./out \
	--output-file-title patient_01 \
	--gate-tree Hits \
	--output-file-format hdf5 \
	--branches-to-extract eventID edep posX posY posZ
```

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | the output file was written |
| `1` | the run failed; the reason is reported on the log |
| `2` | the arguments could not be parsed |

## What Is Checked

- every required option must be given, and the output file title must carry a value
- the input file must exist, end with `.root` and be readable as a ROOT file
- the requested tree must be present; if it is not, the error lists the trees the file holds
- requested branch names must be present in the tree, and must not be empty
- branches whose length varies per entry are reported as unsupported

## Behaviour Worth Knowing

- the output directory is created when missing
- an existing output file is overwritten without a prompt
- the output file holds the extracted tree only; histograms from the input file are not copied
- nothing is written when the run fails

## Running Without Installing The Script

```bash
python -m opengate_gate_tree --help
```
