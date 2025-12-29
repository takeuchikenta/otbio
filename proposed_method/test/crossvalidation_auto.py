import crossvalidation

dir = 'output/garu/otbio_test2_garu_'

crossvalidation.main(dir = dir, subject='garu', feature_name_for_filename = "rms", registration='itk')