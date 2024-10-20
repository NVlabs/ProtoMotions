# Model Overview

## Description:
The model maps multi-modal user commands to actions for a simulated robot in IsaacGym.

This model is for research and development only.

## License/Terms of Use:
See [LICENSE](LICENSE).

## References:
1. [MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inptaining](https://research.nvidia.com/labs/par/maskedmimic/assets/SIGGRAPHAsia2024_MaskedMimic.pdf)
2. [MaskedMimic project page](https://research.nvidia.com/labs/par/maskedmimic/)
3. [MaskedMimic video](https://research.nvidia.com/labs/par/maskedmimic/assets/MaskedMimic.mp4)

## Model Architecture:
Architecture Type: Variational Autoencoder (VAE) with a Transformer encoder. 

**Network Architecture:** Custom

## Input: 
Input Type(s): [text, target poses, target objects] 

**Input Format(s):** [text, motion capture, object]

**Input Parameters:** 
1D [text], 3D [motion capture], 2D [object]

**Other Properties Related to Input:**
Text is represented as X-CLIP embeddings.
We support the SMPL character. A target pose consists of 69 joints. A target pose can contain target positions and rotations. Multiple target poses can be provided in parallel.
A target object is represented by its bounding box, orientation, and 1-hot class indicator. We support 7 object classes. Straight Chairs, Armchairs, Tables, High Stools, Low Stools, Sofas, and Large Sofas.

## Output: 
Output Type(s): Continuous vector 

**Output Format:** Vector

**Output Parameters:** 1D

**Other Properties Related to Output:** The model returns a vector of actions. Each entry corresponds to a single joint in the humanoid robot. 69 dimensions corresponding to the character’s joints. When simulated, these actions control the robotic humanoid. These actions correspond to the SMPL humanoid.

## Model Version(s):
v1.0 , the official release accompanying the SIGGRAPH Asia 2024 MaskedMimic paper. 

# Training, Testing, and Evaluation Datasets:

## Training Dataset:

**Link:** [AMASS](https://amass.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [30 hours of motion capture data from over 300 subjects across over 11000 motion clips. Each clip is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [SAMP](https://samp.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [130 motions and object files. Each motion is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [HML3D](https://github.com/EricGuo5513/HumanML3D) 

** Data Collection Method by dataset

- [Human] 

** Labeling Method by dataset

- [Human] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Labels for the AMASS dataset. Each motion is provided 3 textual labels.]

## Testing Dataset:

**Link:** [AMASS](https://amass.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [30 hours of motion capture data from over 300 subjects across over 11000 motion clips. Each clip is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [SAMP](https://samp.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [130 motions and object files. Each motion is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [HML3D](https://github.com/EricGuo5513/HumanML3D) 

** Data Collection Method by dataset

- [Human] 

** Labeling Method by dataset

- [Human] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Labels for the AMASS dataset. Each motion is provided 3 textual labels.]

## Evaluation Dataset:

**Link:** [AMASS](https://amass.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [30 hours of motion capture data from over 300 subjects across over 11000 motion clips. Each clip is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [SAMP](https://samp.is.tue.mpg.de/) 

** Data Collection Method by dataset

- [Automatic/Sensors] 
- [Human] 

** Labeling Method by dataset

- [Not Applicable] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [130 motions and object files. Each motion is several seconds long. The motion capture data is in the [SMPL](https://smpl.is.tue.mpg.de/) format.]

**Link:** [HML3D](https://github.com/EricGuo5513/HumanML3D) 

** Data Collection Method by dataset

- [Human] 

** Labeling Method by dataset

- [Human] 

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Labels for the AMASS dataset. Each motion is provided 3 textual labels.]

## Inference:
**Engine:** None

**Test Hardware:** NVIDIA A6000 GPU, AMD Ryzen Threadripper PRO 3975WX 32-Cores

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
