#Importing the dependencies

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
import datetime as dt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import pymongo
import urllib.parse
import random as rnd

###region---------------------------------FUNCTIONS---------------------------------------
def prediction(imdb_link,mymodel,myscaler):
    # Opening a browser to scape data from IMDb.
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location=os.environ.get("GOOGLE_CHROME_BIN")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)
    driver.get(imdb_link)
    time.sleep(3)

    # Scraping the elements
    imdb_rating = driver.find_elements_by_class_name('AggregateRatingButton__RatingScore-sc-1ll29m0-1.iTLWoV')
    runtime_year_and_release = driver.find_elements_by_class_name('ipc-inline-list__item')
    genre = driver.find_elements_by_class_name(
        'GenresAndPlot__GenreChip-cum89p-3.fzmeux.ipc-chip.ipc-chip--on-baseAlt')
    movie_name = driver.find_element_by_xpath(
        '//*[@id="__next"]/main/div/section[1]/section/div[3]/section/section/div[1]/div[1]/h1').text

    # IMDB Rating
    imdb_rating = imdb_rating[0].text
    # Year
    for item in range(0, len(runtime_year_and_release)):
        try:
            if item < 5 and int(runtime_year_and_release[item].text):
                year_movie = int(runtime_year_and_release[item].text)
        except:
            continue

    # Runtime
    for item in range(0, len(runtime_year_and_release)):
        try:
            if "h " in runtime_year_and_release[item].text and int(
                    runtime_year_and_release[item].text.split('h')[0]) < 100:
                runtime = runtime_year_and_release[item].text
        except:
            continue

    runtime = runtime.split(' ')
    hour = int(runtime[0].split('h')[0]) * 60
    minute = int(runtime[1].split('m')[0])
    runtime_mins = hour + minute
    runtime_mins = float(runtime_mins)

    # Genres
    genre = genre[0].text
    try:
        genre = genres_dict[genre]
    except:
        genre = 0

    # How Old
    how_old = dt.datetime.now().year - year_movie

    # Month
    mon_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12}

    for item in range(0, len(runtime_year_and_release)):
        for month in list(mon_dict.keys()):
            if month in runtime_year_and_release[item].text:
                month_movie = runtime_year_and_release[item].text

    month_movie = month_movie.split(' ')[0]
    month_movie = mon_dict[month_movie]

    # Number of Votes
    driver.get(imdb_link + 'ratings/?ref_=tt_ov_rt')
    time.sleep(3)
    num_votes = driver.find_elements_by_class_name('allText')
    try:
        num_votes1 = int(num_votes[0].text.split(' ')[0].replace(',', ''))
    except:
        num_votes1 = int(num_votes[0].text.split(' ')[0].replace('.', ''))

    imdb_rating_2=num_votes[0].text.split(' /')[0].split(' ')[-1].replace(',','.')
    driver.close()

    try:
        prob = int(mymodel.predict_proba(
            myscaler.transform([[imdb_rating, runtime_mins, year_movie, num_votes1, genre, how_old, month_movie]]))[0][
                       1] * 100)
    except:
        prob = int(mymodel.predict_proba(
            myscaler.transform([[(imdb_rating_2), runtime_mins, year_movie, num_votes1, genre, how_old, month_movie]]))[0][
                       1] * 100)
    return holder.write(f"With {str(prob)}% probability, you will like {movie_name}!")
###endregion

#----------------------------STREAMLIT UI-----------------------------

st.header("IMDb Movie Recommender")
st.write("This tool helps you to predict if you like a movie.")
placeholder_subheader=st.empty()
placeholder_subheader.subheader("<--- Train your model first!")
choice=st.sidebar.radio("What do you want to do?",("Train New Model","Use Existing Model"))

if choice=="Train New Model":
    file=st.sidebar.file_uploader("Upload your IMDb Ratings File")
    if file is not None:
        df=pd.read_csv(file, encoding = "ISO-8859-1")

    if st.sidebar.button("Train"):
        placeholder = st.empty()
        placeholder.write("Your model is being trained...")
        ###region model training
        df = df[df['Title Type'] == 'movie'] #Only selecting movie types
        df.reset_index(inplace=True, drop=True)
        df.drop(['Const', 'Title', 'URL', 'Directors', 'Title Type'], axis=1, inplace=True)  #Dropping the unrelated
        df["Genres_New"] = df.Genres.str.split(',', expand=True)[0] #Generating a new genres column that consist of the first genres of the genres column
        df.dropna(inplace=True) #Dropping the missing value columns
        df["How_Old"] = (pd.to_datetime(df["Date Rated"]).dt.year - pd.to_datetime(df["Release Date"]).dt.year).astype(
            'int64') #Creating a new column called "How_Old" to see how old movies are mostly liked by the user
        df["Year"] = pd.to_datetime(df["Release Date"]).dt.year #Creating New Columns as Year and Month from the Release Date
        df["Month"] = pd.to_datetime(df["Date Rated"]).dt.month
        df.drop(['Date Rated', 'Release Date'], axis=1, inplace=True) #We can drop Date Rated and Release Date columns
        unique_ratings_liked = df[df["Your Rating"] > 5]["Your Rating"].unique() #In our model, we are going to assume that if user's ratings are higher than 5, it means user liked the movie, else, the user disliked the movie
        unique_ratings_disliked = df[df["Your Rating"] <= 5]["Your Rating"].unique()

        df["Your Rating"] = df["Your Rating"].replace(list(unique_ratings_disliked), 0)
        df["Your Rating"] = df["Your Rating"].replace(list(unique_ratings_liked), 1)

        cols_to_encode = ['Genres_New'] #Converting categorical variables into categorical codes using cat.codes()
        for item in cols_to_encode:
            df[item] = df[item].astype('category')
        df[item] = df[item].cat.codes

        X = df.drop(['Your Rating', 'Genres'], axis=1) #Assigning independent and dependent variables
        y = df['Your Rating']

        scaler = MinMaxScaler() # Scaling X values
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y) #Splitting for test-train sets as 80% train and 20% test.

        # Finding the best neighbor value for KNN

        neighbors = np.arange(1, 9)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        # Loop over different values of k
        for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
            knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

        diff = list(train_accuracy - test_accuracy)
        neighbor_number = diff.index(min(diff)) + 1

        knn = KNeighborsClassifier(n_neighbors=neighbor_number).fit(X_train, y_train)
        y_test_pred_knn = knn.predict(X_test)

        matrix = confusion_matrix(y_test, y_test_pred_knn)

        recall1 = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        precision1 = matrix[1][1] / (matrix[1][1] + matrix[0][1])
        F1_score1 = 2 * recall1 * precision1 / (recall1 + precision1)

        recall0 = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        precision0 = matrix[0][0] / (matrix[0][0] + matrix[1][0])
        F1_score0 = 2 * recall0 * precision0 / (recall0 + precision0)

        #INSERTING THE MODEL TO THE DATABASE

        knn_serial = pickle.dumps(knn) #Serializing the model to insert into the db
        scaler_serial = pickle.dumps(scaler) #Serializing the scaler to insert into the db

        #Creating unique key for the user
        letters = ["a", "b", "c", "e", "g", "h", "o", "p", "q", "s", "u", "v", "z"]
        numbers = ["1", "5", "9", "0", "2", "3", "4", "6"]
        key = ""
        for i in range(0, 10):
            key += letters[rnd.randint(0, len(letters) - 1)]
            key += numbers[rnd.randint(0, len(numbers) - 1)]

        df["Genres"] = df.Genres.str.split(',', expand=True)[0]

        model_record = {'key': key,
                      'model': knn_serial,
                      'scaler': scaler_serial,
                      'genres_keys': df["Genres"].unique().tolist(),
                      'genres_values': df["Genres_New"].unique().tolist()} #Creating the document

        username = urllib.parse.quote_plus(os.environ.get("MONGO_USER_ID"))
        password = urllib.parse.quote_plus(os.environ.get("MONGO_USER_PASS"))
        client = pymongo.MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.olkfo.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        db = client.get_database('imdb_data')  # DB Connection
        records = db.imdb
        records.insert_one(model_record) #Saving the document

        #---RESULTS--

        placeholder_subheader.subheader("<--- Press Use Existing Model and Enter Your Key!")
        placeholder.write("Your model is trained!")
        st.write(f"Accuracy is {int(F1_score1 * 100)}% for the liked movies.")
        st.write(f"Your unique key is: {key}. Save your key.")

elif choice=="Use Existing Model":
    placeholder_subheader.subheader("<---Enter your unique key to be able to use your model!")
    unique_key=st.sidebar.text_input("Enter your key")
    imdb_link=st.text_input("Enter a IMDb link")

    if st.button("Predict"):
        holder=st.empty()
        holder.write("....Predicting....")
        username = urllib.parse.quote_plus(os.environ.get("MONGO_USER_ID"))
        password = urllib.parse.quote_plus(os.environ.get("MONGO_USER_PASS"))
        client = pymongo.MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.olkfo.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        db = client.get_database('imdb_data')  # DB Connection
        records = db.imdb

        myquery = {"key": unique_key}
        mydoc = records.find(myquery)
        for i in mydoc:
            model = i['model']
            scaler = i['scaler']
            genres_keys=i['genres_keys']
            genres_values=i['genres_values']

        mymodel = pickle.loads(model)
        myscaler = pickle.loads(scaler)

        genres_dict = {}

        for i in range(0, len(genres_keys) - 1):
            genres_dict[genres_keys[i]] = genres_values[i]

        try:
            imdb_link = imdb_link.split('?')[0]
        except:
            imdb_link=imdb_link
        prediction(imdb_link,mymodel,myscaler)

