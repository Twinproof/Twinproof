# TwinProof

This repository provides the core implementation of TwinProof, a system for verifying the integrity of indoor location claims based on multimodal environmental consistency.

## 1. Environment Requirements

- Python 3.9
- Windows is recommended.

## 2. Data Format

TwinProof uses multimodal sensing data collected with commercial smartphones, including:

### Cellular Signals

- Sampling rate: approximately 0.5 Hz
- Data: IDs and RSSI values of visible cellular base stations

### Geomagnetic Signals

- Sampling rate: 50 Hz
- Data: three-axis magnetic field strength

### Inertial Measurement Unit (IMU) Data

- Sampling rate: 50 Hz
- Fields:
  - `Acc_X`, `Acc_Y`, `Acc_Z`
  - `Meg`
  - `Gyr_X`, `Gyr_Y`, `Gyr_Z`
  - `Ore`

## 3. Data Preprocessing

Raw sensing data must first be preprocessed. This includes:

- Removing anomalous or incomplete sensing segments
- Synchronizing data from different sensors
- Organizing the data into a format suitable for subsequent twin-space construction and verification

Run the scripts in the `data_preprocessing` folder in sequence to complete data preprocessing.

## 4. Twin-Space Graph Construction

TwinProof first learns the environment-induced state-transition structure from historical trajectories.

The construction process includes the following steps:

### 4.1 State Node Generation

Run the scripts in the `Find_Anchor` folder to perform:

- Multimodal sensing segment extraction
- Stable environmental state discovery
- Twin-space node generation

### 4.2 Transition Relationship Construction

Run the scripts in the `path_reconstruction` folder to recover state-transition relationships from historical trajectories and construct the complete Twin-Space Graph.

## 5. Claim Generation

The `Claim` folder provides methods for generating different types of location claims and attack samples, including:

- Legitimate location claims
- Direct forgery attacks
- Replay attacks
- Trajectory transplantation attacks
- Remote proxy attacks

Users can generate the appropriate verification samples for their needs.

## 6. Claim Integrity Detection

The `Claim_Detection` folder contains TwinProof's complete integrity verification modules.

Different detection workflows are categorized by filename. Users can select and run the appropriate module according to their experimental requirements.

## Data and Experiment Reproduction

### Data Privacy

The raw collected data may contain information that reveals the actual deployment area. Therefore, the complete raw dataset is not publicly released in this repository.

We provide anonymized sample data to demonstrate the data format, processing workflow, and how to run the code.

### Collecting Your Own Data

TwinProof allows users to collect data in their own environments for experimentation.

Sensing data that meets the format requirements above can be used to construct a twin-space graph for the corresponding environment. In general, a relatively stable representation of the environmental states can be obtained after traversing the target area along approximately 20 or more trajectories.

### Code Availability

This repository provides the core algorithmic implementation of TwinProof.

To facilitate rapid deployment and testing, default values are provided for some parameters. For detailed experimental parameters and the complete configuration, please refer to the paper.
