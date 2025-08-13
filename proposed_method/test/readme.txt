# Open Access Dataset and Toolbox of High-Density Surface Electromyogram Recordings

We provide an open access dataset of High densitY Surface Electromyogram (HD-sEMG) Recordings (named "Hyser"), and a toolbox (implemented using Matlab, availabe on GitHub (https://github.com/Open-EMG/toolbox)) for neural interface research. Our dataset consisted of data from 20 subjects (8 female and 12 male volunteers). All subjects signed a written informed consent. This study was reviewed and approved by the ethics committee of Fudan University (approval number: BE2035).

This Hyser dataset contains five sub-datasets as: (1) pattern recognition (PR) dataset acquired during 34 commonly used hand gestures, (2) maximal voluntary muscle contraction (MVC) dataset while subjects contracted each individual finger, (3) one-degree of freedom (DoF) dataset acquired during force-varying contraction of each individual finger, (4) N-DoF dataset acquired during prescribed contractions of combinations of multiple fingers, and (5) random task dataset acquired during random contraction of combinations of fingers without any prescribed force trajectory. Dataset 1 can be used for gesture recognition studies. Datasets 2--5 also recorded individual finger forces, thus can be used for studies on proportional control of neuroprostheses. 

Gesture labels of PR dataset were saved in ``*.txt" files with comma-separated values format. All force trajectory waveforms and EMG signal waveforms were saved in waveform database (WFDB) format, with one ``*.dat" file storing all 16-bit signed type quantitized values, and one ``*.hea" file storing the scaling factors and other supplementary information. The 256-channel HD-sEMG were acquired by four 8*8 electrode arrays. The electrode arrays were named as "ED" (extensor-distal), "EP" (extensor-proximal), "FD" (flexor-distal) and "FP" (flexor-proximal), placing on the distal end of extensor muscle, the proximal end of extensor muscle, the distal end of flexor muscle and the proximal end of the flexor muscle, respectively. Signal name of each sEMG channel was named by "XX-i-j", where "XX" is one of "ED", "EP", "FD" and "FP", to indicate the applied electrode array. Both "i" and "j" are integers ranging from 1 to 8, to indicate the index of row and column in the electrode array. Signal name of each force channel was named by "thumb", "index", "middle", "ring" or "little" to indicate the force of a specific finger.

Our toolbox can be used to: (1) analyze each of the five datasets using standard benchmark methods and (2) decompose HD-sEMG signals into motor unit action potentials via independent component analysis. We expect our dataset and toolbox can provide a unique platform to promote a wide range of neural interface research and collaboration among neural rehabilitation engineers.

Please see   https://physionet.org/   for more details on our Hyser dataset.

License:
Open Data Commons Attribution License v1.0

Contact:
Dr. Chenyun Dai
School of Information Science and Technology, 
Fudan University, Shanghai 200433, China.
chenyundai@fudan.edu.cn
