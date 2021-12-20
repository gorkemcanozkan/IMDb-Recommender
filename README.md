# IMDb-Recommender

Link: http://imdbrecommender.herokuapp.com/

This project aims to help users learn if they will like a movie or not. The system learns from the user's imdb rating data. Then, when the user enters the link of a movie that they might watch, the system predicts whether the movie is likeable by the user by showing a percentage.

# Libraries Used

- Selenium
- Scikit-learn
- Streamlit
- PyMongo
- NumPy
- Pandas

# Step-by-Step Guide

1) If you have a IMDB account (and if you have rated 50+ movies so far), go to "Your Ratings" page, then click on the 3 dots on the top right, and export.

![image](https://user-images.githubusercontent.com/71969715/146769242-8b682842-4a2a-44b1-89b5-52388fd6a114.png)

2) Go to http://imdbrecommender.herokuapp.com/

3) From the menu on the left, click browse files and select the "ratings.csv" file that you have downloaded from imdb.com

![image](https://user-images.githubusercontent.com/71969715/146769436-4dfef938-83e0-421a-9b61-76ba17d1f9b8.png)

4) Click on "Train" button. This is the machine learning part. The system learns from your previous data by using the information of the movies, and the rating that you gave.

5) When the training is completed, the system will return the accuracy score, and also your unique key for the model. Your model and scaler gets saved to the database as binary strings, and you can query them by using your key anytime. Save this key to a notepad file for the prediction stage.

![image](https://user-images.githubusercontent.com/71969715/146769807-91674d10-2823-41cc-bd77-18e9b780d620.png)

6) From the top left, select "Use Existing Model".

![image](https://user-images.githubusercontent.com/71969715/146770002-0cfa805f-7cda-48ce-a88a-ab3a10eaceb3.png)

7) Enter your unique key here for the prediction. This will help you access your own model (also the scaler).

8) Then enter the movie link that you want to watch and click "Predict".

![image](https://user-images.githubusercontent.com/71969715/146770466-29037e3a-cc71-4365-b106-f70c00f3e76b.png)

9) It is going to take around 1 minute to scrape the data from the imdb link and predicts the outcome.

![image](https://user-images.githubusercontent.com/71969715/146770545-61abeff9-80fb-47ff-812d-d094c48ad29b.png)




