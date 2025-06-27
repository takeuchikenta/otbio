import OTBiolabInterface as otb
import OTBiolabClasses as otbClasses
import numpy as np
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt

''' DESCRIPTION 
'''

''' CATEGORY
ProposedMethod
'''



############################################## PARAMETERS #########################################################

#var1=...
#var2=...
#var3=...
samplerate=2000
a=48
b=52
c=49
d=51
gpass=3
gstop=40

###################################################################################################################


############################################# LOADING DATA ########################################################

#Use one function from otb library to load data from main software
tracks=otb.LoadDataFromPythonFolder()

###################################################################################################################



############################################## ALGORITHM ##########################################################
#Develope your code here

#Example function for data elaboration. This example just report as output the input values
def Example_Function(samples, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "bandstop")       #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, samples)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す
	
    
    
#Main Code - Example code to show how elaborate tracks and save them. Change the elaboration with your own code
result_tracks=[]
for track in tracks:
	result_sections=[]
	number_of_channels=0
	
	for section in track.sections:
		
		result_channels=[]
		for channel in section.channels:
			
			
			elaborated_data=Example_Function(channel.data, samplerate=samplerate, fp=np.array([a,b]), fs=np.array([c,d]), gpass=gpass, gstop=gstop) #channel.data contains your samples
			result_channels.append(otbClasses.Channel(elaborated_data))
		
		number_of_channels=len(result_channels)
		result_sections.append(otbClasses.Section(section.start, section.end, result_channels))
	
	result_tracks.append(otbClasses.Track(result_sections, track.frequency, number_of_channels,unit_of_measure=track.unit_of_measure , title='Example - '+track.title))



###################################################################################################################




############################################ WRITE DATA ###########################################################

#Use one function from otb library to plot data to main software or to continue the processing chain
otb.WriteDataInPythonFolder(result_tracks)

###################################################################################################################
