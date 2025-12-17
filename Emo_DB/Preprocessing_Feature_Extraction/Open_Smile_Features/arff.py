import os
import shlex
import subprocess
from typing import List
import shutil

'''
Python Datei zum Extrahieren von Feature Sets von OpenSmile für den MSP Podcast Corpus. 
Hintergrund: Ich hatte eGeMAPS für die Emo DB extrahiert und anschließend ein h5 Datensatz erstellt. 
Ich war mit den Klassifizierungsergebnissen aber nicht zufrieden und bin dann bei MFCC geblieben.
Ich wollte dies auch für den MSP-Podcast machen, jedoch hat die Zeit und mein Speicherplatz nicht ausgereicht.
'''

def run_smile_extract(config_path: str, file_path: str, save_path: str):
    try:
        command = f"SMILExtract -C {config_path} -I {file_path} -O {save_path[:-4]}.arff"
        subprocess.run(shlex.split(command), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error {e}")


def main(audio_data_path_msp: str, save_folder_path: str, config_path: str):
    if not os.path.isdir(audio_data_path_msp) or not os.path.isdir(save_folder_path):
        print("Error no directory.")
        return

    list_audio_paths: List[str] = os.listdir(audio_data_path_msp)

    wav_files = [file for file in list_audio_paths if file.endswith('.wav')]

    for file in wav_files:
        file_path = os.path.join(audio_data_path_msp, file)
        save_path = os.path.join(save_folder_path, file)
        
        if os.path.isfile(file_path):
            run_smile_extract(config_path, file_path, save_path)


if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(main_path, "config/compare16/ComParE_2016.conf")
    audio_data_path_msp = os.path.join(main_path, "../../data/MSP_Podcast_Corpus/Audio/")
    save_folder_path = os.path.join(main_path, "../../data/MSP_Podcast_Corpus/arff_files/")
    shutil.rmtree(save_folder_path)
    os.makedirs(save_folder_path)
    main(audio_data_path_msp, save_folder_path, config_path)
