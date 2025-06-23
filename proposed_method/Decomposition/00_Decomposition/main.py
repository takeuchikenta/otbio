import OTBiolabInterface as otb
import OTBiolabClasses as otbClasses

''' DESCRIPTION 
MUAP decomposition
'''

''' CATEGORY
ProposedMethod
'''



############################################## PARAMETERS #########################################################

#var1=...
#var2=...
#var3=...

###################################################################################################################


############################################# LOADING DATA ########################################################

#Use one function from otb library to load data from main software
tracks=otb.LoadDataFromPythonFolder()

###################################################################################################################



############################################## ALGORITHM ##########################################################
#Develope your code here

#Example function for data elaboration. This example just report as output the input values
def Example_Function(samples):
	
    #Samples elaboration
	
    return samples
    
    
#Main Code - Example code to show how elaborate tracks and save them. Change the elaboration with your own code
result_tracks=[]
for track in tracks:
	result_sections=[]
	number_of_channels=0
	
	for section in track.sections:
		
		result_channels=[]
		for channel in section.channels:
			
			
			elaborated_data=Example_Function(channel.data) #channel.data contains your samples
			result_channels.append(otbClasses.Channel(elaborated_data))
		
		number_of_channels=len(result_channels)
		result_sections.append(otbClasses.Section(section.start, section.end, result_channels))
	
	result_tracks.append(otbClasses.Track(result_sections, track.frequency, number_of_channels,unit_of_measure=track.unit_of_measure , title='Example - '+track.title))



###################################################################################################################




############################################ WRITE DATA ###########################################################

#Use one function from otb library to plot data to main software or to continue the processing chain
otb.WriteDataInPythonFolder(result_tracks)

###################################################################################################################
