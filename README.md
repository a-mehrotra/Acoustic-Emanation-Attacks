# Acoustic Emanation Attacks
<br />
The goal of this lab was to see how physical side channels like acoustic data could leak information. The attack implemented analyzes an audio recording of keyboard typings to steal a secret. The inference was conducted with a Neural Network trained by preprocessing a set of audio recordings for each character. 
<br />
See pdf for results <br />
The AcousticAttack.py was run with python version 3.9.2 <br />
Before running, update the following directory locations: <br />
Update the sys.path.insert directory to point to the location of extractKeyStroke.py if the file is located in another directory. <br />
Update the location of the variable 'f' in get_KeyPress_TrainingData to the same directory where the AcousticAttack.py file is stored on your local machine. <br />
Update the audio_data variable in test_Attack_Password function and generate_MLP_Model function to point to the data folder on your local machine <br />
<br />
To run AcousticAttack.py, use the following command in the terminal: <br />
python3 AcousticAttack.py

