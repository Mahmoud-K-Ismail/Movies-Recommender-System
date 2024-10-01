# Movies Recommender System 

Movies Recommender System is a comprehensive project that implements various recommendation algorithms to suggest movies to users based on their preferences and ratings. This system utilizes collaborative filtering (both user-user and item-item) and content-based filtering techniques to predict user preferences and recommend movies.

## Features

- **Collaborative Filtering**: Utilizes user-user and item-item collaborative filtering to recommend movies.
- **Content-Based Filtering**: Uses movie topics and user reviews to compute taste vectors and make recommendations.
- **Customizable Recommendations**: Allows users to get recommendations based on different filtering methods and similarity measures.
- **Interactive GUI**: Includes a user-friendly graphical interface for inputting user data and displaying recommended movies.

## Models

- **User-User Collaborative Filtering**: Predicts user preferences by finding similar users and averaging their ratings.
- **Item-Item Collaborative Filtering**: Recommends movies by finding similar items based on user ratings.
- **Content-Based Filtering**: Recommends movies based on the content similarity between movies and user preferences.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Pandas
- NumPy
- SciPy
- Matplotlib

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/movies-recommender-system.git
   cd movies-recommender-system
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset: Ensure the datasets are available in the project directory. This includes:
   - `non_personalised_stereotyped_rec.csv`
   - `content_based_filtering.csv`
   - `User-User Collaborative Filtering - movie-row.csv`
   - `Item Item Collaborative Filtering - Ratings.csv`

### Running the Model

1. Load and preprocess the data:
   ```sh
   python preprocess.py
   ```
2. Train the collaborative filtering models:
   ```sh
   python train_collaborative.py
   ```
3. Train the content-based filtering model:
   ```sh
   python train_content.py
   ```

### Generating Recommendations

1. Run the interactive script:
   ```sh
   python recommend.py
   ```
2. Enter user details and preferences to get movie recommendations.

### Usage

You can use the interactive script to get movie recommendations or integrate the models into other applications. The GUI allows for easy input of user data and viewing of recommended movies.

## Example
   ```python
   from recommender import generate_recommendations

   user_id = 1
   recommendations = generate_recommendations(user_id, num_recommendations=5)
   print(recommendations)
   ```

### Contributing

We welcome contributions! Please read our Contributing Guidelines for more information on how to contribute to this project.

### Acknowledgments

- Project Gutenberg for providing the dataset.
- Pandas and NumPy for data manipulation.
- SciPy for scientific computing tools.
