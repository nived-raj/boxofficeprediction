import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


import joblib


# Load the pickled KNN model
knn_model = joblib.load('movie_revenue_model.pkl')


#import the orginal file
movies_df_augmented = pd.read_csv('mov_dat_stream.csv')

#For the input datafame column -
df_columns = ['Action',
 'Adventure',
 'Animation',
 'Biography',
 'Comedy',
 'Crime',
 'Drama',
 'Family',
 'Fantasy',
 'History',
 'Horror',
 'Music',
 'Musical',
 'Mystery',
 'Romance',
 'Sci-Fi',
 'Sport',
 'Thriller',
 'War',
 'Western',
 'Aamir Khan',
 'Al Pacino',
 'Albert Brooks',
 'Alexandra Maria Lara',
 'Amy Adams',
 'Anthony Hopkins',
 'Arnold Schwarzenegger',
 'Ben Affleck',
 'Ben Kingsley',
 'Benicio Del Toro',
 'Bill Murray',
 'Bill Nighy',
 'Bill Paxton',
 'Billy Crudup',
 'Brad Pitt',
 'Bradley Cooper',
 'Brenda Blethyn',
 'Brendan Gleeson',
 'Bruce Willis',
 'Bryan Cranston',
 'Carrie Fisher',
 'Cary Elwes',
 'Cate Blanchett',
 'Catherine Keener',
 'Chiwetel Ejiofor',
 'Chloë Grace Moretz',
 'Chris Cooper',
 'Chris Evans',
 'Chris Hemsworth',
 'Chris Pine',
 'Chris Pratt',
 'Christian Bale',
 'Christopher Lloyd',
 'Christopher Plummer',
 'Christopher Walken',
 'Clint Eastwood',
 'Clive Owen',
 'Colin Firth',
 'Dakota Fanning',
 'Daniel Craig',
 'Daniel Day-Lewis',
 'Daniel Radcliffe',
 'Danny Aiello',
 'Danny Glover',
 'Denzel Washington',
 'Diane Keaton',
 'Diane Kruger',
 'Dianne Wiest',
 'Domhnall Gleeson',
 'Donald Sutherland',
 'Dustin Hoffman',
 'Ed Harris',
 'Edward Norton',
 'Elijah Wood',
 'Emma Stone',
 'Emma Watson',
 'Ethan Coen',
 'Ethan Hawke',
 'Ewan McGregor',
 'Frances McDormand',
 'Franka Potente',
 'Gael García Bernal',
 'Gary Oldman',
 'Gena Rowlands',
 'Gene Hackman',
 'Gene Wilder',
 'Geoffrey Rush',
 'George Clooney',
 'Gerard Butler',
 'Guy Pearce',
 'Harrison Ford',
 'Hugh Jackman',
 'Ian McKellen',
 'Irrfan Khan',
 'Jack Nicholson',
 'Jack Warden',
 'Jake Gyllenhaal',
 'James Caan',
 'James McAvoy',
 'Jared Leto',
 'Javier Bardem',
 'Jay Baruchel',
 'Jean-Louis Trintignant',
 'Jeff Bridges',
 'Jennifer Connelly',
 'Jeremy Renner',
 'Joaquin Phoenix',
 'Jodie Foster',
 'Joe Pesci',
 'Joe Russo',
 'John Cazale',
 'John Goodman',
 'John Malkovich',
 'John Musker',
 'John Turturro',
 'Johnny Depp',
 'Jon Bernthal',
 'Joseph Gordon-Levitt',
 'Josh Brolin',
 'Jude Law',
 'Judi Dench',
 'Julianne Moore',
 'Julie Delpy',
 'Kang-ho Song',
 'Kate Winslet',
 'Kathy Bates',
 'Keira Knightley',
 'Ken Watanabe',
 'Kerry Washington',
 'Kevin Bacon',
 'Kevin Costner',
 'Kevin Spacey',
 'Kurt Russell',
 'Laura Dern',
 'Lee Unkrich',
 'Leonardo DiCaprio',
 'Liam Neeson',
 'Mads Mikkelsen',
 'Marion Cotillard',
 'Mark Hamill',
 'Mark Ruffalo',
 'Mark Wahlberg',
 'Martin Freeman',
 'Matt Damon',
 'Matthew Broderick',
 'Matthew McConaughey',
 'Mel Gibson',
 'Meryl Streep',
 'Mia Farrow',
 'Michael Caine',
 'Michael Madsen',
 'Michelle Williams',
 'Morgan Freeman',
 'Naomi Watts',
 'Natalie Portman',
 'Nawazuddin Siddiqui',
 'Nicolas Cage',
 'Nicole Kidman',
 'Octavia Spencer',
 'Orlando Bloom',
 'Patricia Arquette',
 'Philip Seymour Hoffman',
 'Rachel McAdams',
 'Ralph Fiennes',
 'Robert De Niro',
 'Robert Downey Jr.',
 'Robert Duvall',
 'Robin Williams',
 'Robin Wright',
 'Rooney Mara',
 'Roy Scheider',
 'Rupert Grint',
 'Russell Crowe',
 'Ryan Gosling',
 'Sam Rockwell',
 'Samantha Morton',
 'Samuel L. Jackson',
 'Sandra Bullock',
 'Scarlett Johansson',
 'Sean Connery',
 'Sean Penn',
 'Shah Rukh Khan',
 'Sigourney Weaver',
 'Simon Pegg',
 'Stellan Skarsgård',
 'Steve Carell',
 'Sushant Singh Rajput',
 'Sylvester Stallone',
 'Tabu',
 'Taika Waititi',
 'Tim Allen',
 'Tim Robbins',
 'Tom Cruise',
 'Tom Hanks',
 'Tom Hardy',
 'Tony Chiu-Wai Leung',
 'Uma Thurman',
 'Val Kilmer',
 'Viggo Mortensen',
 'Vin Diesel',
 'Willem Dafoe',
 'Winona Ryder',
 'Won Bin',
 'Woody Allen',
 'Woody Harrelson',
 'Zoe Saldana',
 'Director_Alan Parker',
 'Director_Alejandro Amenábar',
 'Director_Alejandro G. Iñárritu',
 'Director_Alfonso Cuarón',
 'Director_Ang Lee',
 'Director_Anthony Russo',
 'Director_Asghar Farhadi',
 'Director_Bong Joon Ho',
 'Director_Brad Bird',
 'Director_Brian De Palma',
 'Director_Bryan Singer',
 'Director_Chan-wook Park',
 'Director_Christopher Nolan',
 'Director_Clint Eastwood',
 'Director_Danny Boyle',
 'Director_Darren Aronofsky',
 'Director_David Fincher',
 'Director_David Lynch',
 'Director_David Yates',
 'Director_Denis Villeneuve',
 'Director_Edgar Wright',
 'Director_Edward Zwick',
 'Director_Francis Ford Coppola',
 'Director_Giuseppe Tornatore',
 'Director_Guy Ritchie',
 'Director_Hayao Miyazaki',
 'Director_J.J. Abrams',
 'Director_James Cameron',
 'Director_James Mangold',
 'Director_Jim Jarmusch',
 'Director_Jim Sheridan',
 'Director_Joel Coen',
 'Director_John Hughes',
 'Director_John McTiernan',
 'Director_Krzysztof Kieslowski',
 'Director_Lars von Trier',
 'Director_Martin Scorsese',
 'Director_Matthew Vaughn',
 'Director_Mel Gibson',
 'Director_Michael Mann',
 'Director_Paul Greengrass',
 'Director_Paul Thomas Anderson',
 'Director_Pedro Almodóvar',
 'Director_Pete Docter',
 'Director_Peter Jackson',
 'Director_Quentin Tarantino',
 'Director_Rajkumar Hirani',
 'Director_Richard Linklater',
 'Director_Ridley Scott',
 'Director_Rob Reiner',
 'Director_Robert Zemeckis',
 'Director_Roman Polanski',
 'Director_Ron Clements',
 'Director_Ron Howard',
 'Director_Sam Mendes',
 'Director_Satoshi Kon',
 'Director_Sidney Lumet',
 'Director_Spike Jonze',
 'Director_Spike Lee',
 'Director_Stanley Kubrick',
 'Director_Steven Spielberg',
 'Director_Taika Waititi',
 'Director_Terry Gilliam',
 'Director_Tim Burton',
 'Director_Tom McCarthy',
 'Director_Wes Anderson',
 'Director_Woody Allen',
 'Certificate_A',
 'Certificate_G',
 'Certificate_PG',
 'Certificate_PG-13',
 'Certificate_R',
 'Certificate_U',
 'Certificate_UA',
 'Epic (> 180 min)',
 'Long Feature (120-150 min)',
 'Short Feature (60-90 min)',
 'Standard Feature (90-120 min)',
 'Very Long Feature (150-180 min)']

 # Create a dataframe where all values are 0 initially
input_df = pd.DataFrame(0, index=[0], columns=df_columns)





# Sample lists for dropdown options
directors = [
    'Alan Parker', 'Alejandro Amenábar', 'Alejandro G. Iñárritu', 'Alfonso Cuarón', 
    'Ang Lee', 'Anthony Russo', 'Asghar Farhadi', 'Bong Joon Ho', 'Brad Bird', 
    'Brian De Palma', 'Bryan Singer', 'Chan-wook Park', 'Christopher Nolan', 
    'Clint Eastwood', 'Danny Boyle', 'Darren Aronofsky', 'David Fincher', 
    'David Lynch', 'David Yates', 'Denis Villeneuve', 'Edgar Wright', 'Edward Zwick', 
    'Francis Ford Coppola', 'Giuseppe Tornatore', 'Guy Ritchie', 'Hayao Miyazaki', 
    'J.J. Abrams', 'James Cameron', 'James Mangold', 'Jim Jarmusch', 'Jim Sheridan', 
    'Joel Coen', 'John Hughes', 'John McTiernan', 'Krzysztof Kieslowski', 'Lars von Trier', 
    'Martin Scorsese', 'Matthew Vaughn', 'Mel Gibson', 'Michael Mann', 'Paul Greengrass', 
    'Paul Thomas Anderson', 'Pedro Almodóvar', 'Pete Docter', 'Peter Jackson', 
    'Quentin Tarantino', 'Rajkumar Hirani', 'Richard Linklater', 'Ridley Scott', 
    'Rob Reiner', 'Robert Zemeckis', 'Roman Polanski', 'Ron Clements', 'Ron Howard', 
    'Sam Mendes', 'Satoshi Kon', 'Sidney Lumet', 'Spike Jonze', 'Spike Lee', 
    'Stanley Kubrick', 'Steven Spielberg', 'Taika Waititi', 'Terry Gilliam', 
    'Tim Burton', 'Tom McCarthy', 'Wes Anderson', 'Woody Allen'
]

actors = [
    'Aamir Khan', 'Al Pacino', 'Albert Brooks', 'Alexandra Maria Lara', 'Amy Adams', 
    'Anthony Hopkins', 'Arnold Schwarzenegger', 'Ben Affleck', 'Ben Kingsley', 
    'Benicio Del Toro', 'Bill Murray', 'Bill Nighy', 'Bill Paxton', 'Billy Crudup', 
    'Brad Pitt', 'Bradley Cooper', 'Brenda Blethyn', 'Brendan Gleeson', 'Bruce Willis', 
    'Bryan Cranston', 'Carrie Fisher', 'Cary Elwes', 'Cate Blanchett', 'Catherine Keener', 
    'Chiwetel Ejiofor', 'Chloë Grace Moretz', 'Chris Cooper', 'Chris Evans', 'Chris Hemsworth', 
    'Chris Pine', 'Chris Pratt', 'Christian Bale', 'Christopher Lloyd', 'Christopher Plummer', 
    'Christopher Walken', 'Clint Eastwood', 'Clive Owen', 'Colin Firth', 'Dakota Fanning', 
    'Daniel Craig', 'Daniel Day-Lewis', 'Daniel Radcliffe', 'Danny Aiello', 'Danny Glover', 
    'Denzel Washington', 'Diane Keaton', 'Diane Kruger', 'Dianne Wiest', 'Domhnall Gleeson', 
    'Donald Sutherland', 'Dustin Hoffman', 'Ed Harris', 'Edward Norton', 'Elijah Wood', 
    'Emma Stone', 'Emma Watson', 'Ethan Coen', 'Ethan Hawke', 'Ewan McGregor', 'Frances McDormand', 
    'Franka Potente', 'Gael García Bernal', 'Gary Oldman', 'Gena Rowlands', 'Gene Hackman', 
    'Gene Wilder', 'Geoffrey Rush', 'George Clooney', 'Gerard Butler', 'Guy Pearce', 'Harrison Ford', 
    'Hugh Jackman', 'Ian McKellen', 'Irrfan Khan', 'Jack Nicholson', 'Jack Warden', 'Jake Gyllenhaal', 
    'James Caan', 'James McAvoy', 'Jared Leto', 'Javier Bardem', 'Jay Baruchel', 'Jean-Louis Trintignant', 
    'Jeff Bridges', 'Jennifer Connelly', 'Jeremy Renner', 'Joaquin Phoenix', 'Jodie Foster', 
    'Joe Pesci', 'Joe Russo', 'John Cazale', 'John Goodman', 'John Malkovich', 'John Musker', 
    'John Turturro', 'Johnny Depp', 'Jon Bernthal', 'Joseph Gordon-Levitt', 'Josh Brolin', 
    'Jude Law', 'Judi Dench', 'Julianne Moore', 'Julie Delpy', 'Kang-ho Song', 'Kate Winslet', 
    'Kathy Bates', 'Keira Knightley', 'Ken Watanabe', 'Kerry Washington', 'Kevin Bacon', 
    'Kevin Costner', 'Kevin Spacey', 'Kurt Russell', 'Laura Dern', 'Lee Unkrich', 'Leonardo DiCaprio', 
    'Liam Neeson', 'Mads Mikkelsen', 'Marion Cotillard', 'Mark Hamill', 'Mark Ruffalo', 
    'Mark Wahlberg', 'Martin Freeman', 'Matt Damon', 'Matthew Broderick', 'Matthew McConaughey', 
    'Mel Gibson', 'Meryl Streep', 'Mia Farrow', 'Michael Caine', 'Michael Madsen', 'Michelle Williams', 
    'Morgan Freeman', 'Naomi Watts', 'Natalie Portman', 'Nawazuddin Siddiqui', 'Nicolas Cage', 
    'Nicole Kidman', 'Octavia Spencer', 'Orlando Bloom', 'Patricia Arquette', 'Philip Seymour Hoffman', 
    'Rachel McAdams', 'Ralph Fiennes', 'Robert De Niro', 'Robert Downey Jr.', 'Robert Duvall', 
    'Robin Williams', 'Robin Wright', 'Rooney Mara', 'Roy Scheider', 'Rupert Grint', 
    'Russell Crowe', 'Ryan Gosling', 'Sam Rockwell', 'Samantha Morton', 'Samuel L. Jackson', 
    'Sandra Bullock', 'Scarlett Johansson', 'Sean Connery', 'Sean Penn', 'Shah Rukh Khan', 
    'Sigourney Weaver', 'Simon Pegg', 'Stellan Skarsgård', 'Steve Carell', 'Sushant Singh Rajput', 
    'Sylvester Stallone', 'Tabu', 'Taika Waititi', 'Tim Allen', 'Tim Robbins', 'Tom Cruise', 
    'Tom Hanks', 'Tom Hardy', 'Tony Chiu-Wai Leung', 'Uma Thurman', 'Val Kilmer', 'Viggo Mortensen', 
    'Vin Diesel', 'Willem Dafoe', 'Winona Ryder', 'Won Bin', 'Woody Allen', 'Woody Harrelson', 
    'Zoe Saldana'
]

genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
          'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 
          'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']

certificates = [
    'A', 'G', 'PG', 'PG-13', 'R', 'U', 'UA'
]

runtime_categories = [
    'Epic (> 180 min)', 
    'Long Feature (120-150 min)', 
    'Short Feature (60-90 min)', 
    'Standard Feature (90-120 min)', 
    'Very Long Feature (150-180 min)'
]


# Function to update the dataframe based on user input
def update_input_df(input_df, selected_director, selected_actor1, selected_actor2, selected_actor3, 
                    selected_genres, selected_certificate, selected_runtime):
    """
    Updates the input dataframe based on the user inputs. Sets the appropriate column values to 1.
    """
    # Update Director (with prefix)
    director_column = f'Director_{selected_director}'
    if director_column in input_df.columns:
        input_df[director_column] = 1

    # Update Actors (no prefix, just the name as column)
    for actor in [selected_actor1, selected_actor2, selected_actor3]:
        if actor in input_df.columns:
            input_df[actor] = 1
    
    # Update Genres (no prefix, can be multiple)
    for genre in selected_genres:
        if genre in input_df.columns:
            input_df[genre] = 1
    
    # Update Certificate (with prefix)
    certificate_column = f'Certificate_{selected_certificate}'
    if certificate_column in input_df.columns:
        input_df[certificate_column] = 1
    
    # Update Runtime (no prefix, just the name as column)
    if selected_runtime in input_df.columns:
        input_df[selected_runtime] = 1
    
    return input_df







# Streamlit app to capture user input
st.title("Movie Revenue Predictor: Estimate Box Office Success Based on Your Movie’s Features.")
st.subheader("Have a movie idea in mind? Ever wonder how it would perform at the box office? Just enter the details below and we'll predict it for 2024!")


st.markdown("""
**Fun Fact:** You can also check classic movies and see how they would gross in today's market! we have included top actors and directors from 1970 so you can choose and experiment!
""")

# Add a horizontal line
st.markdown("---")

# User inputs
selected_director = st.selectbox('Select Director', directors)


# Create three columns for the actor dropdowns
actor_col1, actor_col2, actor_col3 = st.columns(3)

# Add each actor dropdown to its own column, excluding already selected actors
selected_actor1 = actor_col1.selectbox('Select Actor 1', actors)
available_actors_for_actor2 = [actor for actor in actors if actor != selected_actor1]
selected_actor2 = actor_col2.selectbox('Select Actor 2', available_actors_for_actor2)

available_actors_for_actor3 = [actor for actor in actors if actor not in {selected_actor1, selected_actor2}]
selected_actor3 = actor_col3.selectbox('Select Actor 3', available_actors_for_actor3)



# Create three columns for the genre dropdowns
genre_col1, genre_col2, genre_col3 = st.columns(3)

# Add each genre dropdown to its own column
selected_genre1 = genre_col1.selectbox('Select Genre 1', genres)
available_genres_for_genre2 = ['None'] + [g for g in genres if g != selected_genre1]
selected_genre2 = genre_col2.selectbox('Select Genre 2', available_genres_for_genre2)

available_genres_for_genre3 = ['None'] + [g for g in genres if g not in {selected_genre1, selected_genre2}]
selected_genre3 = genre_col3.selectbox('Select Genre 3', available_genres_for_genre3)


# Create two columns for certificate and runtime dropdowns
cert_col, runtime_col = st.columns(2)

# Add the certificate and runtime dropdowns to their respective columns
selected_certificate = cert_col.selectbox('Select Certificate', certificates)
selected_runtime = runtime_col.selectbox('Select Runtime', runtime_categories)


# Add a button for submitting the inputs
if st.button('Predict'):
    # Data validation for unique actor selections
    if len({selected_actor1, selected_actor2, selected_actor3}) != 3:
        st.error("Please select unique actors for Actor 1, Actor 2, and Actor 3.")
    else:
        # Ensure at least one genre is selected and genres are unique
        selected_genres = {selected_genre1, selected_genre2, selected_genre3} - {'None'}  # Filter out 'None'

        if len(selected_genres) < 1:
            st.error("Please select at least one genre.")
        else:
            # Display the selected inputs when the button is clicked

            st.write(f"{selected_certificate} rated {'-'.join(selected_genres)} movie directed by {selected_director} with {selected_actor1}, {selected_actor2} and {selected_actor3} in lead roles which is a {selected_runtime} is expected to bag:")

            
            # Update input_df based on the selected inputs
            input_df = update_input_df(input_df, selected_director, selected_actor1, selected_actor2, 
                                       selected_actor3, selected_genres, selected_certificate, selected_runtime)
            
            # Convert user input to model input
            user_input = input_df.values
            
            # Step 3: Predict with the user input and find the closest movie
            distances, indices = knn_model.kneighbors(user_input)

            # Step 4: Get the Adjusted_Gross of the nearest movie
            closest_movies = movies_df_augmented.iloc[indices[0]]
            nearest_movie_gross = closest_movies['Adjusted_Gross'].values[0]

            # Step 5: Scale the gross based on the distance
            distance_to_nearest = distances[0][0]  # Cosine distance to the nearest movie

            # If cosine distance is close to 0, movies are very similar (close to 1 if dissimilar)
            # Scale the predicted gross based on how similar the movies are
            scaling_factor = max(0.8, (1 - distance_to_nearest))  # Don't scale below 80%
            predicted_gross = nearest_movie_gross * scaling_factor
            predicted_gross_rounded = round(predicted_gross, -3)

            # Display predicted gross in a sleek money-green color
            st.markdown(f"""
            <h1 style='color:#28a745; font-size:50px;'>
                ${predicted_gross_rounded:,.2f}
            </h1>
            """, unsafe_allow_html=True)

            

            # Display the predicted gross revenue
            st.write(f"(Predicted Gross Revenue for 2024 - rounded to nearest 1000)")

            

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .footer {
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        text-align: center;
        color: grey;
        font-size: 12px;
        padding: 10px;
    }
    @media (max-width: 768px) {
        .footer {
            font-size: 10px; /* Smaller font size for mobile */
        }
    }
    </style>
    <div class="footer">
        <p> Disclaimer: This model is currently in its initial development phase, utilizing a dataset of 1000 movies. Efforts are underway to further enhance and scale the dataset for improved accuracy and performance. <br>
        Created by <b>Nived Raj</b></p>
    </div>
    """, unsafe_allow_html=True)




