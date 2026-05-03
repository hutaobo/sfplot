# CellGPS User Guide and Distribution Notes

## 1. Executive Summary

`CellGPS.exe` is a Windows x64 single-file GUI application packaged from the `sfplot` codebase.

For end users, it behaves like a standalone desktop program:

- users do not need to install Python or Conda
- users can launch the software by double-clicking a single `.exe`
- the application is suitable for distribution to external researchers on Windows

Technically, it is a PyInstaller-style one-file build, which means the executable unpacks its runtime files into the user's temporary directory before the GUI starts.

This is still a valid standalone distribution model, but it is important to describe it accurately:

- it is standalone for normal Windows users
- it is not a pure native binary with zero runtime assumptions

## 2. Build Characteristics

The current official release has the following characteristics:

- target platform: `Windows 64-bit`
- subsystem: `Windows GUI`
- packaging style: `single-file executable`
- official release file: `CellGPS.exe`
- official release size: about `297 MB`

This release prioritizes stability and compatibility for the CSV and Xenium workflows over aggressive size reduction.

## 3. Release Status

The distributed executable should be treated as the official release build:

- the release filename is `CellGPS.exe`
- older experimental or intermediate packaging variants should not be distributed
- the current release was finalized after compatibility-focused packaging adjustments for the Xenium workflow

## 4. Standalone Verdict

The executable can be distributed as a standalone Windows application, provided that the target machine meets these conditions:

- Windows 10 or Windows 11
- 64-bit operating system
- permission to write to the user's `%TEMP%` directory
- no security product blocking execution from the executable or its temporary extraction directory

In practical terms, most researchers should be able to download the file, copy it to a local folder, and run it without installing extra software.

## 5. Known Limitations

Please keep the following limitations in mind when distributing the application:

1. The executable still depends on temporary extraction during startup.
   First launch may be slower than a typical native desktop app.

2. Some corporate or institutional antivirus products may flag or delay one-file scientific Python executables.

3. The GUI currently focuses on loading data, computing results, and displaying heatmaps.
   It is not yet a full export-oriented desktop workflow.

4. In Xenium mode, if only `analysis.tar.gz` is available, the software may extract analysis files into the selected Xenium folder.
   That folder therefore needs write permission.

## 6. Recommended Distribution Practice

For external release, distribute the software with:

- `CellGPS.exe`
- this guide in `docx` or `pdf` format

When sharing the executable with researchers, use wording like this:

> CellGPS is a Windows 64-bit single-file desktop build that already contains the required Python runtime and major dependencies. In most cases, no additional software installation is needed. The first launch may take longer because the application unpacks its runtime files into the user's temporary directory.

## 7. System Requirements

- Windows 10 or Windows 11
- 64-bit system
- recommended memory: at least `8 GB RAM`
- permission to read input datasets and write to the local temporary directory

## 8. Before Launching the Program

### 8.1 Copy the executable to a local disk

Do not run the program directly from:

- a network drive
- an email attachment preview
- a compressed archive

Recommended locations:

- `C:\Users\<username>\Desktop\CellGPS\CellGPS.exe`
- `C:\Users\<username>\Downloads\CellGPS\CellGPS.exe`

### 8.2 Unblock the file if Windows marked it as downloaded

If the file was downloaded from a browser, cloud drive, or email:

1. Right-click `CellGPS.exe`
2. Open `Properties`
3. If an `Unblock` option is visible, enable it
4. Click `Apply` and `OK`

### 8.3 Security software may intervene

On managed institutional machines, Windows SmartScreen or endpoint security software may delay or block the first launch.

If this happens, the user may need to:

- allow the application to run
- or ask their IT/security team to whitelist it

## 9. Launch Instructions

To start the application:

1. Double-click `CellGPS.exe`
2. Wait for the GUI to finish loading

Notes:

- first startup can take `10-30` seconds depending on the machine
- startup may be slower than ordinary desktop software because of temporary extraction
- if no window appears immediately, wait before assuming the launch failed

## 10. GUI Overview

The application currently contains two workflows:

- `CSV Heatmap`
- `Xenium Heatmap`

## 11. CSV Heatmap Workflow

### 11.1 Required input

This mode expects a CSV file with at least these columns:

- `x`
- `y`
- `celltype`

If the file uses different column names, the current GUI does not provide a column-mapping dialog, and the workflow will fail.

### 11.2 Steps

1. Open the program
2. Go to the `CSV Heatmap` tab
3. Click `Select CSV File`
4. Choose the input CSV file
5. Click `Plot CSV Heatmap`
6. Wait while the application reads the file, computes distances, and renders the heatmap
7. View the result in the display panel
8. Adjust `Zoom` if needed

### 11.3 Output behavior

The current GUI displays the result inside the application window.

At the moment, there is no dedicated export button in the GUI, so users should treat the current build primarily as an analysis-and-preview interface.

## 12. Xenium Heatmap Workflow

### 12.1 Expected Xenium folder structure

The selected Xenium folder should contain enough information for the loader to recover clustering and UMAP data through one of these paths:

- pre-existing `analysis/.../clusters.csv` and `analysis/.../projection.csv`
- `analysis.tar.gz`
- `analysis.h5`

### 12.2 Required selection CSV

The second CSV file must contain a column named exactly:

- `Cell ID`

The program uses this column to select cells from the loaded Xenium dataset.

### 12.3 Steps

1. Open the program
2. Go to the `Xenium Heatmap` tab
3. Click `Select Xenium Dir`
4. Select the Xenium data folder
5. Click `Load Xenium Data`
6. Wait until loading finishes
7. Click `Select Selection CSV`
8. Choose the CSV file that contains the `Cell ID` column
9. Click `Plot Xenium Heatmap`
10. Wait for the result to appear in the display panel

### 12.4 Important note about write access

If only `analysis.tar.gz` is present and the extracted analysis folder is missing, the program may unpack analysis files into the selected Xenium directory.

That means:

- the folder must be writable
- the software may modify the data directory by adding extracted analysis content

If users want to avoid changing the original dataset folder, they should work on a local copy.

## 13. Troubleshooting

### 13.1 Nothing appears when double-clicking

Check these first:

- the file is being run from a local disk, not a network path
- Windows did not block the file
- security software did not quarantine or suspend the process
- the program has been given enough time to finish first-time extraction

### 13.2 The program opens and then fails

Check whether an `error.log` file was created next to the executable.

This is the main runtime error log for the packaged build.

### 13.3 Can researchers on other computers run it?

In most cases, yes, if they are using:

- Windows x64
- a machine where `%TEMP%` is writable
- valid input data in the expected format

### 13.4 Why is the file still large?

Because it still bundles a substantial scientific Python runtime and data processing stack required for heatmap generation and Xenium loading.

The official release is a compatibility-focused scientific desktop executable rather than a minimal native utility.

## 14. Recommended Final Validation Before Public Release

Before publishing globally, run one final smoke test on a clean Windows x64 machine that does not have your local Python or Conda development setup.

Recommended checklist:

1. Copy only `CellGPS.exe` to the machine
2. Launch the executable
3. Run a small `CSV Heatmap` example
4. Run a small `Xenium Heatmap` example
5. Confirm that no unexpected `error.log` is created
6. Confirm that startup time is acceptable

## 15. Final Assessment

The official `CellGPS.exe` is suitable for distribution to external researchers as a standalone Windows GUI application.

It should be described as:

- a single-file Windows x64 scientific desktop application
- with bundled runtime dependencies
- requiring no separate Python installation for normal users

That is an accurate and defensible description for global distribution.
