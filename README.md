# RaagIndentificationModel

AI driven Classical Raag Recognition System
By Rahul Purandare, Sai Bhujbal and Tanay Kende

## Introduction:

The word “raag” comes from the Sanskrit word ”ragam”  which means color or hue. Music or any form of art has many shades to it and each shade has a distinct essence. What gives any song its essence? You listen to the song “Dekha ek khaab” from the film “Silsila” and it feels like a light breeze of love that sways you away. On the other hand, if you listen to the iconic “Ae mere watan ke logon”   by Lata Mangeshkar, your heart fills with patriotism and gratitude for martyrs. And the melody of Madhubala’s “Pyar kiya toh darna kya”  makes u feel the power of defiant love.
Apart from the lyrics, what makes these songs portray these emotions? It is the framework of the notes that produce this effect. This framework of notes that is properly defined by certain rules of swar progression and patterns is called a raag. There are more than 200 ragas in Hindustani Classical Muisc. Each raag has a nuanced pattern of notes and hence produces a different effect. According to Bharata’s Natyashastra written by Bharata in the 2nd Century CE, there are nine rasas  or essences. These are called navarasa and they encapsulate nine different emotions like anger, disgust, compassion, happiness etc. Each raag expresses any one or more of these emotions and thus we feel the corresponding emotions after listening to the raags.

## Objective:
AI as we know has brought monumental change in every sphere of life. It has made many operations easy in several domains like architecture, finance, defense, media etc. Similarly, it has brought path breaking developments in the field of music. Ragas are characterized by the patterns of notes that are played/sung. Identifying ragas based on these patterns and classifying using machine learning was an interesting project that we sought to pursue. The commercial applications of this project include genre identification of a song, training classical music students for raag identification and on raag therapy that many psychiatrists suggest helps improve temperament.
Although there are 200 and above raags in Hindustani Classical Music, we chose to work on only 10 basic raags, which are Bhoop, Durga, Bhairav, Bhairavi, Yaman, Malkans, Puriya dhanashree, Bhimplasi, Alhaiya Bilawal and Asavari. Each raag has a distinct set of notes and fixed patterns. We have specifically chosen these raags as they are easily distinguishable because of their unique patterns.

## Methodology:

 
- **Data Collection:**
The first step was to gather audio files of the 20 ragas we had chosen for classification. We chose five audio files of each raag of length of about 40 mins to 1 hour 15 mins. These were the performances of stalwarts like Pt. Bhimsen Joshi, Parveen Sultana, Kishori Amonkar etc. We also added our self-created data as one of our team members, Rahul Purandare has been pursuing classical music for the last 10 years. We arranged the audio files into subfolders of each raag. The audio files were of the format .mp3 or .m4a.

- **Preprocessing:**
Preprocessing was arguably the most difficult part of the project. We had to train a CNN model that required equal number of 5 sec audio files for each raag. We trimmed the first two and last two minutes of each audio files to avoid disturbances like mic setting and applauses using AudioSegment module of the Pydub library.. So we segmented our main audio files into 30 minutes each and hence got 5 audio files of each raag. For raags where our audio files were more than 1 hour in length we got two segments and hence considered only 4 audio files. Then we performed standard preprocessing functions like noise reduction (removing unwanted frequencies from the audio file) and normalization(controlling the volume of the files) using the noisereduce library. Then we went for the more challenging part, which was pitch conversion. We wanted to train our model is C# scale but all our audio files were not of the same pitch. So using the librosa library’s pitch shift function, we transformed the pitch of each audio file by changing the semitone value. For example for an audio file of pitch A#, the value of the semitone change was 3 as C# is 3 semitiones above A#. After changing the pitch of all the files, we further segmented them into 5 seconds to suit the working of our CNN model.

- **Feature Extraction:**
Feature extraction is the process of deriving features of an audio file. To draw inferences from an Audio file, there are several aspects of it that need to be taken into consideration like its frequency, amplitude etc. There are many ways of extracting features from audio files like MFCCs(Mel frequency cepstral coefficients), Chroma Features, Spectrogram, Centroid, Zero crossing rate etc. We decided to work on MFCCS. MFCCs or Mel frequency cepstral coefficient are a cluster of features extracted from an audio file that display its spectral characteristics. They aim to mimic human auditory system’s response to audio by capturing essential aspects of the signal. Using Librosa library. We extract mfcc values of each of our audio files. The values were obtained in an array format representing the expanse of the audio file. Given below is the plot of a typical mfcc extraction of an audio file.
 

- **Model Training:**
We have trained our model using Convolution Neural Network. The model was designed using three convolution layers, using Adam optimizer, learning rate of 0.0001, relu as activation function and using softmax function to produce a probabilitstic classification at the output layer. The source code for this CNN was keras. We got a accuracy of 71.6% using this model.

- **Deployment:**
The code was deployed using streamlit library. We created an interface with the user. We ask the user to input an audio file of the format .wav,.mp3 or .m4a and then the pitch of that audio file is identified. Following this, the pitch is converted to C#. And then features are extracted and subsequently the raag is predicted.

- **Learning Outcome:**
By this project, we learned to handle audio files and extract features out of them. Typically, audio file handling is not as easy as handling modalities like text, numbers but getting insights on it helped us link our love for music to the concepts the data science. We further want to work on expanding the scope of AI related projects in the field of classical music so that the legacy of classical music and its beauty is understood by all.

