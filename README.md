# million_song_dataset
An exploration and analysis of the Million Song Dataset using Pyspark and collaborative filtering recommender systems. 

For this task, we will be exploring a collection of datasets known as the Million Song Dataset (MSD), which started as a collaborative project between The Echo Nest and LabROSA.

The main dataset contains the song ID, the track ID, the artist ID, and 51 other fields, such as the year, title, artist tags, and various audio properties such as loudness, beat, tempo, and time signature. Note that track ID and song ID are not the same concepts - the track ID corresponds to a particular recording of a song, and there may be multiple (almost identical) tracks for the same song. Tracks are the fundamental identifier and are matched to songs. Songs are then matched to artists as well.

The Million Song Dataset also contains other datasets contributed by organisations and the community,
* SecondHandSongs (cover songs)
* musiXmatch dataset (song lyrics)
* Last.fm dataset (song-level tags and similarity)
* Taste Profile subset (user-song plays)
* thisismyjam-to-MSD mapping (user-song plays, imperfectly joined)
* tagtraum genre annotations (genre labels)
* Top MAGD dataset (more genre labels)

During this task, we will be focusing on Taste Profile, Audio Features and MSD AllMusic Genre Dataset (MAGD).

We will begin by preprocessing the data and making necessary joins using Pyspark. We will then analyse audio similarity and build a collaborative filtering model to give song recommendations.

