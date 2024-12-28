# from itertools import count
import plotly.graph_objects as go
from PIL import Image
from nltk.corpus import stopwords
from scipy.optimize import direct
from selenium.common import TimeoutException
# from requests import options
from selenium.webdriver import ActionChains
import streamlit as st
# from time import sleep
# from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
# import time
import numpy as np
# import tensorflow
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
from textblob import Word

chrome_options = Options()
chrome_options.add_argument('--headless')  # Uncomment if you want to run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--window-size=1920,1080')
# ________________________________________________________________________________________________________________________

# important downloads
import nltk

nltk.download('omw-1.4')  # For multilingual WordNet support
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# ________________________________________________________________________________________________________________________

stop_words = stopwords.words('english')

model_file_path = 'my_sentiment_model.pkl'
tokenizer_file_path = 'tokenizer.pkl'


# ________________________________________________________________________________________________________________________

with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# ________________________________________________________________________________________________________________________

# Initialize session state for navigation and inputs

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "username" not in st.session_state:
    st.session_state.username = ""
if "password" not in st.session_state:
    st.session_state.password = ""

# Initialize session state
if "urls" not in st.session_state:
    st.session_state.urls = []

if "rankings" not in st.session_state:
    st.session_state.rankings = []

if "credentials_ok" not in st.session_state:
    st.session_state.credentials_ok = False

if "compute_signal" not in st.session_state:
    st.session_state.compute_signal = False

# ________________________________________________________________________________________________________________________

def is_amazon_product_url(url):
    # Regular expression to check if the URL is an Amazon product page
    amazon_pattern = r"(https?://(?:www\.)?amazon\.[a-z]+/.*?/dp/\w+)"
    return re.match(amazon_pattern, url) is not None


# ________________________________________________________________________________________________________________________

def amazon_login(driver, username, password):
    """
    Logs into Amazon using the provided credentials and returns a logged-in WebDriver instance.

    Args:
        username (str): Amazon username or email.
        password (str): Amazon password.

    Returns:
        WebDriver: Selenium WebDriver instance logged into Amazon.
        str: Message indicating success or failure.
    """
    try:
        # Initialize the Selenium WebDriver (adjust driver initialization as needed)

        # Click on the sign-in link
        # sign_in_link = WebDriverWait(driver, 10).until(
        #     EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        # )
        # sign_in_link.click()

        # Enter username
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_email"))
        )
        username_field.send_keys(username)
        username_field.submit()

        # Enter password
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        password_field.send_keys(password)
        password_field.submit()

        # Verify successful login
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )

        # Check if the text inside the account link has changed (indicating login success)
        if account_link.text != "Hello, sign in":
            return driver, "Login Successful"
        else:
            return None, "Login Failed"

    except Exception as e:
        print(f"Error during login: {e}")
        return None, "Login Failed"

# ________________________________________________________________________________________________________________________
def amazon_logout(driver):
    """
    Logs out from Amazon using the provided logged-in WebDriver instance.

    Args:
        driver (WebDriver): A Selenium WebDriver instance logged into Amazon.

    Returns:
        str: Message indicating success or failure.
    """
    try:
        # Hover over the account link to reveal the dropdown menu
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        actions = ActionChains(driver)
        actions.move_to_element(account_link).perform()

        # Wait for the sign-out button to appear and be clickable
        sign_out_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "nav-item-signout"))
        )
        sign_out_button.click()

        # Verify logout by checking if "Hello, sign in" text appears
        sign_in_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        if sign_in_link.text == "Hello, sign in":
            return "Logout Successful"
        else:
            return "Logout Failed"

    except Exception as e:
        print(f"Error during logout: {e}")
        return "Logout Failed"

# ________________________________________________________________________________________________________________________

def check_amazon_credentials(username, password):
    try:
        # Initialize progress bar
        progress_bar = st.progress(0)
        progress = 0

        # Navigate to Amazon login page
        driver = webdriver.Chrome()
        driver.get("https://www.amazon.com/")
        progress += 20
        progress_bar.progress(progress)

        # Click on the sign-in link
        sign_in_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        sign_in_link.click()
        progress += 20
        progress_bar.progress(progress)

        # Enter username
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_email"))
        )
        username_field.send_keys(username)
        username_field.submit()
        progress += 20
        progress_bar.progress(progress)

        # Enter password
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        password_field.send_keys(password)
        password_field.submit()
        progress += 20
        progress_bar.progress(progress)

        # Wait for account link to confirm login success
        account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
        )
        progress += 10
        progress_bar.progress(progress)

        if account_link.text != "Hello, sign in":
            # Hover over the account link for the logout process
            actions = ActionChains(driver)
            actions.move_to_element(account_link).perform()

            # Wait for the sign-out button
            sign_out_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#nav-item-signout > span"))
            )
            sign_out_button.click()
            progress += 10
            progress_bar.progress(progress)

            # Confirm logout if required
            try:
                confirm_logout_button = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "auth-action-sign-out-form-submit"))
                )
                confirm_logout_button.click()
            except:
                pass  # No confirmation required

            progress_bar.progress(100)
            return "Login Successful"

        # Handle login failure
        try:
            error_message = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "auth-error-message-box"))
            ).text
            if "Invalid username or password" in error_message:
                st.error("Invalid Username or Password")
                return "Invalid Username or Password"
            else:
                st.error("Login Failed")
                return "Login Failed"
        except:
            st.error("Login Failed")
            return "Login Failed"

    except Exception as e:
        st.error(f"Error during login or logout")
        return "Login Failed"

# ________________________________________________________________________________________________________________________

def scrape_reviews(url, num):
    product_details = {"product_name": "", "reviews": [], "stars": []}
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)  # Wait for up to 10 seconds for elements to load
    st.write(f'Progress of URL {num}')
    progress_bar = st.progress(0)  # Initialize the progress bar

    try:
        driver.get(url)

        # Wait for the product title to load, throw an exception if not found in 10 seconds
        try:
            product_title_element = wait.until(EC.presence_of_element_located((By.ID, "productTitle")))
            if product_title_element is not None:
                product_details['product_name'] = product_title_element.text
        except TimeoutException:
            st.error("Timed out waiting for the product title.")
            driver.quit()
            return product_details  # Return empty details if the title is not found

        # Navigate to "See All Reviews"
        try:
            reviews_link = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@data-hook="see-all-reviews-link-foot"]')))
            reviews_link.click()
        except TimeoutException:
            st.error("Timed out waiting for the reviews link.")
            driver.quit()
            return product_details  # Return empty details if the reviews link is not clickable

        amazon_login(driver, st.session_state.username, st.session_state.password)

        page = 1
        total_pages = 10  # Set this based on the number of pages you expect, or calculate dynamically

        while True:
            html_data = BeautifulSoup(driver.page_source, 'html.parser')

            review_divs = html_data.find_all('div', {'id': re.compile(r'^customer_review-')})

            # Extract reviews, ratings, and metadata
            for review in review_divs:
                review_body = review.find('span', {'data-hook': 'review-body'})
                if review_body:
                    product_details['reviews'].append(review_body.get_text(strip=True))
                rating_element = review.find('i', {'data-hook': 'review-star-rating'})
                if rating_element:
                    rating_text = rating_element.find('span', {'class': 'a-icon-alt'}).text
                    product_details['stars'].append(rating_text.split()[0])

            # Check for next page
            next_page = html_data.find('li', {'class': 'a-last'})
            if not next_page or 'a-disabled' in next_page.get('class', ''):
                break

            try:
                next_page_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//ul[@class='a-pagination']//li[@class='a-last']/a"))
                )
                next_page_button.click()
                wait.until(EC.staleness_of(next_page_button))
                page += 1
            except TimeoutException:
                st.error("Reviews are not accessible!!")
                return -1

            # Update progress bar
            progress_bar.progress(page / total_pages)  # Update the progress bar

        amazon_logout(driver)
        progress_bar.progress(100)
        driver.quit()

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        progress_bar.progress(100)
        driver.quit()

    return product_details

# ________________________________________________________________________________________________________________________

def modifyReviews(reviews_array):
    """
    Preprocesses an array of input reviews to match the format used during training.

    Args:
        reviews_array (list): A list of reviews to be preprocessed.

    Returns:
        numpy.ndarray: The preprocessed reviews as a padded sequence.
    """
    # Ensure input is a list of strings
    processed_reviews = []
    for review in reviews_array:
        # Convert to lowercase and split
        review = ' '.join(review.lower() for review in review.split())

        # Replacing special characters using re.sub()
        review = re.sub(r"[^a-zA-Z\s]", '', review)  # Keep only letters and spaces

        # Replacing digits/numbers using re.sub()
        review = re.sub(r'\d+', '', review)  # Remove all digits

        # Removing emojis using re.sub()
        review = re.sub(r'[^\x00-\x7F]+', '', review)  # Remove non-ASCII characters (including emojis)

        # Removing stop words
        review = ' '.join(review for review in review.split() if review not in stop_words)

        # Lemmatization
        review = ' '.join([Word(word).lemmatize() for word in review.split()])

        processed_reviews.append(review)

    # Convert to sequence and pad
    X = tokenizer.texts_to_sequences(processed_reviews)
    X = pad_sequences(X, maxlen=103)

    return X
# ________________________________________________________________________________________________________________________

def get_scores(reviews_array):
    prediction = model.predict(reviews_array)
    return prediction

# ________________________________________________________________________________________________________________________

# Streamlit UI Configuration
st.set_page_config(
    page_title="AMAZON AI PICKER",
    page_icon=":chart_with_upwards_trend:",
)

st.header("AMAZON AI PICKER")

# Navigation buttons with consistent width
if st.sidebar.button("Home", key="home"):
    st.session_state.page = "Home"
if st.sidebar.button("Credentials", key="credentials"):
    st.session_state.page = "Credentials"
if st.sidebar.button("Model Implementation", key="page2"):
    st.session_state.page = "Model Implementation"
if st.sidebar.button("Results", key="page3"):
    st.session_state.page = "Results"

# ________________________________________________________________________________________________________________________
def display_home_page():
    st.title("🎉 Welcome to **Amazon Review Analyzer**")
    st.subheader("Your go-to tool for analyzing and comparing Amazon product reviews!")

    # Add an engaging introductory message
    st.markdown("""
    ### 📖 **Overview**  
    With **Amazon Review Analyzer**, you can:  
    - **Securely verify your Amazon credentials** in real time.  
    - Analyze up to **4 Amazon products** simultaneously.  
    - Leverage a **state-of-the-art LSTM model**, trained on over **100,000 reviews**, to calculate sentiment scores.  
    - Compare products based on **real customer feedback** and make informed decisions with confidence.  
    """)

    # Key Features Section with a visually appealing layout
    st.markdown("### 📌 **Key Features**")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/d/de/Amazon_icon.png",
                 width=100)  # Replace with a relevant image if necessary
    with col2:
        st.markdown("""
        - **Real-time Credential Validation**: Ensure your Amazon account details are correct.  
        - **URL Validation**: Verify the correctness of product links before analysis.  
        - **Sentiment Analysis**: Get comprehensive sentiment scores for product reviews using a cutting-edge **LSTM model**.  
        - **Ranking System**: Compare and rank products based on review sentiments to make smarter purchase decisions.  
        """)

    # Add a dropdown for "Get Started"
    st.markdown("### 🚀 **Get Started**")
    with st.expander("Expand to Begin"):
        st.markdown("""
        1. **Credentials Page**: Enter your Amazon username and password for secure validation.  
        2. **Model Implementation Page**: Add up to 4 Amazon product links for analysis.  
        3. **Results Page**: See detailed rankings and sentiment analysis for the products.  
        """)
        st.markdown("""
        Ready to explore? Use the navigation menu to get started!
        """)

    # LSTM Model Section
    st.markdown("### 🤖 **About the LSTM Model**")
    st.write("""
    The **Long Short-Term Memory (LSTM)** model is a type of recurrent neural network (RNN) designed to handle sequential data effectively.  
    - Trained on over **100,000 Amazon reviews**, it identifies patterns and context in text, delivering precise sentiment analysis.  
    - By analyzing sentiments in customer reviews, the model enables accurate product ranking based on user feedback.  
    """)
    # Add a section for LSTM model accuracy
    st.markdown("### 📈 **LSTM Model Accuracy**")
    st.write("The following graph shows the accuracy of our LSTM model during training:")

    # Load the image from a local path
    img_path = "Screenshot 2024-12-28 at 4.54.02 PM.png"  # Update this with your image file path
    image = Image.open(img_path)

    # Display the image
    st.image(image, caption="LSTM Model Accuracy Over Time", use_container_width=True)
    # GitHub Link Section
    st.markdown("### 📂 **GitHub Repository**")
    st.write("""
    Explore the complete codebase or contribute to this open-source project:  
    [**GitHub Repository**](https://github.com/Rishabh-441/Sentiment-Analysis-python-notebook/tree/main)
    """)

    # Why Choose This Tool Section
    st.markdown("### 🌟 **Why Choose Amazon Review Analyzer?**")
    st.markdown("""
    - 🚀 **Fast**: Quick results in just minutes.  
    - 🔒 **Secure**: Your data privacy is our priority.  
    - 🤖 **Intelligent**: Powered by advanced machine learning models.  
    - 🎯 **Accurate**: Delivers insights you can rely on for smarter decisions.  
    """)

    # Add a motivational footer
    st.markdown("---")
    st.markdown("""
    👨‍💻 Developed with ❤️ to help you make smarter choices.  
    **Happy analyzing!**  
    """)

# ________________________________________________________________________________________________________________________

# Display the home page
if st.session_state.page == "Home":
    display_home_page()
# ________________________________________________________________________________________________________________________
elif st.session_state.page == "Credentials":
    st.title("Enter your credentials")

    def direct_to_page2():
        st.session_state.credentials_ok = True
        st.session_state.page = "Model Implementation"
        return

    # Text input fields
    username_input = st.text_input(
        "Enter your username:",
        value=st.session_state.username,  # Set value from session state
        placeholder="Type here..."
    )
    password_input = st.text_input(
        "Enter your password:",
        value=st.session_state.password,  # Set value from session state
        placeholder="Type here...",
        type='password'
    )

    # Update session state dynamically
    if username_input != st.session_state.username:
        st.session_state.username = username_input
    if password_input != st.session_state.password:
        st.session_state.password = password_input

    # Submit button
    if st.button("Submit"):
        check = check_amazon_credentials(st.session_state.username, st.session_state.password)
        if check == "Login Successful":
            st.success("Correct Username & Password")
            st.button("Go to Model Implementation", on_click=direct_to_page2)
        else:
            st.error(check)

    # Clear credentials button
    if st.button("Clear Credentials"):
        # Clear session state for username and password
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.credentials_ok = False
        st.success("Credentials have been cleared from the Session Memory.")
# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Model Implementation") & (st.session_state.credentials_ok == True):
    st.title("Model Implementation")

    def direct_to_page3():
        st.session_state.page = "Results"
        return

    # Initialize session state if it's not already present with 4 empty strings
    if "urls" not in st.session_state:
        st.session_state.urls = []

    if "show_predict_button" not in st.session_state:
        st.session_state.show_predict_button = False

    def clear_url_data(index):
        st.session_state.urls[index] = ""

    def clean_urls():
        seen_urls = set()
        for i, url in enumerate(st.session_state.urls):
            if url in seen_urls and url != "":  # Check if the URL is a duplicate
                st.warning(f"Product URL {i + 1} is duplicate!")
                clear_url_data(i)
            if (url != "") &  (not is_amazon_product_url(url)):
                st.warning(f"Product URL {i+1} is not a valid AMAZON Product link!")
                clear_url_data(i)
            elif url != "":  # Add unique, non-empty URLs to the set
                seen_urls.add(url)

        # Filter out empty strings from the URLs list
        urls = list(set(st.session_state.urls))
        st.session_state.urls = list(filter(lambda x: x != "", urls))
        st.write(st.session_state.urls)
        st.session_state.show_predict_button = True


    # Create a single form to handle all input fields
    with st.form(key="url_form"):
        # Loop through and create 4 text input fields for URLs
        for i in range(4):
            st.session_state.urls.append("")
            # Set value for the URL text input
            st.session_state.urls[i] = st.text_input(f"Product URL {i + 1}:", value=st.session_state.urls[i],
                                                     key=f"product_url_{i}")

        # Submit button to submit the form
        submit_button = st.form_submit_button("Submit")

    # If the form is submitted, clean the URLs
    if submit_button:
        clean_urls()

    # Button to trigger the prediction (calls the clean_urls function)
    if st.session_state.show_predict_button:
        if st.button("Predict Ranking"):
            st.session_state.compute_signal = True
            st.button("Show Results ➡", on_click=direct_to_page3)

# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Model Implementation") & (st.session_state.credentials_ok == False):
    st.title("Model Implementation")
    st.subheader("Please enter and verify your credentials!!")
# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Results") & (st.session_state.credentials_ok == True):
    st.title("Results")

    def get_good_urls(urls):
        # Filter out non-empty, stripped Amazon product URLs
        good_urls = [url.strip() for url in urls if url.strip() and is_amazon_product_url(url.strip())]
        return good_urls


    urls = list(set(get_good_urls(st.session_state.urls)))
    st.session_state.urls = list(filter(lambda x: x != "", urls))

    total_urls = len(st.session_state.urls)

    for idx, url in enumerate(st.session_state.urls, start=1):
        st.markdown(f"**{idx}. [Product Link :  {idx}]({url})**")  # Display URLs as clickable links with numbering
    st.markdown("---")

    st.markdown("## Product Details")

    # Check if product_details is already computed and stored
    if ('product_details' not in st.session_state) | st.session_state.compute_signal:
        product_details = []

        # Process each URL and store the results in product_details
        for i, url in enumerate(st.session_state.urls):
            # Simulate or replace this with your actual scraping function
            # product_details.append(scrape_reviews(url))  # Replace with actual function call
            product_dict = scrape_reviews(url, i+1)
            if product_dict == -1:
                product_dict = {
                    "product_name" : -1
                }
            product_dict['url'] = url
            product_details.append(product_dict)  # This will fetch and process the reviews for each URL


        # Store the processed data in session state to avoid re-computation
        st.session_state.product_details = product_details
        st.session_state.compute_signal = False
    else:
        # Use previously computed results from session state
        product_details = st.session_state.product_details

    # Present the product details for each product in a nice format
    for details in product_details:
        if details['product_name'] == -1:
            st.subheader("Product details not found")
            st.write(f"Product link : {details['url']}")
            st.markdown("---")
            continue

        # Create a layout with columns to organize the content
        col1, col2 = st.columns([2, 1])

        avg_star_rating = 0
        # Display the product name in the first column
        with col1:
            st.subheader(details["product_name"])  # Display product name as a header

        with col2:
            st.markdown("### 🌟 Star Ratings Breakdown:")
            if "5.0" in details['stars']:
                st.write("5 ⭐️ : ", details['stars'].count("5.0"))
                avg_star_rating += 5*details['stars'].count("5.0")
            if "4.0" in details['stars']:
                st.write("4 ⭐️ : ", details['stars'].count("4.0"))
                avg_star_rating += 4 * details['stars'].count("4.0")
            if "3.0" in details['stars']:
                st.write("3 ⭐️ : ", details['stars'].count("3.0"))
                avg_star_rating += 3 * details['stars'].count("3.0")
            if "2.0" in details['stars']:
                st.write("2 ⭐️ : ", details['stars'].count("2.0"))
                avg_star_rating += 2 * details['stars'].count("2.0")
            if "1.0" in details['stars']:
                st.write("1 ⭐️ : ", details['stars'].count("1.0"))
                avg_star_rating += 1 * details['stars'].count("1.0")

        avg_star_rating /= max(1,len(details['stars']))
        # Display the average rating if it's not a dictionary
        st.markdown(f"### 🌟 Average Rating: **{round(avg_star_rating,1)}** Stars")


        # Expandable section to show reviews
        with st.expander("Customer Reviews"):
            # Display reviews inside an expandable section for better UI
            for review in details["reviews"]:
                st.write(f"- {review}")
        st.markdown("---")

    sentiment_scores = []
    product_no = 1
    # getting sentiment scores
    for details in product_details:
        product_reviews = details['reviews']
        modified_product_reviews = modifyReviews(product_reviews)

        if len(modified_product_reviews) != 0:
            product_review_scores = get_scores(modified_product_reviews)

            # Convert to a NumPy array if not already one
            product_review_scores = np.array(product_review_scores)

            # Calculate the average of the first and second elements
            avg_first = np.mean(product_review_scores[:, 0])  # Average of the first column
            avg_second = np.mean(product_review_scores[:, 1])  # Average of the second column

            sentiment_scores.append({
                "product_no": product_no,
                "name": details['product_name'],
                "avg_first": avg_first,
                "avg_second": avg_second,
                "url": details['url']
            })
        else:
            sentiment_scores.append({
                "product_no": product_no,
                "name": details['product_name'],
                "avg_first": 0,
                "avg_second": 0,
                "url": details['url']
            })
        product_no += 1

    # Rank products based on avg_second (higher is better)
    ranked_products = sorted(sentiment_scores, key=lambda x: x["avg_second"], reverse=True)

    st.markdown("# Product Rankings")
    for rank, product in enumerate(ranked_products, start=1):
        st.markdown(f"**Rank {rank}: {product['name']}**")
        st.markdown(f" - Average Negative Score: {product['avg_first']:.2f}")
        st.markdown(f" - Average Positive Score: {product['avg_second']:.2f}")

        # Display the product URL as a clickable link
        st.markdown(f" - [Buy here]({product['url']})")

    st.markdown("---")

    for details in product_details:
        if details['product_name'] == -1:
            continue

        star_counts = {
            "5 Stars": details['stars'].count("5.0"),
            "4 Stars": details['stars'].count("4.0"),
            "3 Stars": details['stars'].count("3.0"),
            "2 Stars": details['stars'].count("2.0"),
            "1 Star": details['stars'].count("1.0"),
        }

        fig = go.Figure(data=[go.Pie(labels=list(star_counts.keys()), values=list(star_counts.values()))])
        fig.update_layout(title=f"Star Rating Distribution for {details['product_name']}")
        st.plotly_chart(fig)

    # Helper function to abbreviate product names
    def abbreviate_name(name, max_length=15):
        return name if len(name) <= max_length else name[:max_length] + "..."

    # Abbreviate product names
    product_names = [abbreviate_name(product['name']) for product in ranked_products]
    positive_scores = [product['avg_second'] for product in ranked_products]
    negative_scores = [product['avg_first'] for product in ranked_products]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=product_names, y=positive_scores, name='Positive Sentiment', marker_color='green'))
    fig.add_trace(go.Bar(x=product_names, y=negative_scores, name='Negative Sentiment', marker_color='red'))

    fig.update_layout(
        title="Sentiment Scores of Products",
        xaxis_title="Products",
        yaxis_title="Sentiment Scores",
        barmode='group',
        width=900,  # Set the desired width
        height=500,  # Set the desired height
        title_font=dict(size=20),  # Increase title font size
    )

    st.plotly_chart(fig)

    st.markdown("---")

    # Iterate over each product's details
    for details in product_details:
        # Skip if product details are not found or there are no reviews
        if details['product_name'] == -1 or len(details['reviews']) == 0:
            continue

        # Get sentiment scores for reviews
        review_scores = get_scores(modifyReviews(details['reviews']))  # Assume scores are in chronological order

        # Create a DataFrame for sentiment scores
        df = pd.DataFrame(review_scores, columns=["Negative", "Positive"])

        # Create a new figure for each product
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["Positive"], mode='lines+markers', name='Positive Sentiment', line_color='green'))
        fig.add_trace(go.Scatter(y=df["Negative"], mode='lines+markers', name='Negative Sentiment', line_color='red'))

        # Update layout for the figure
        fig.update_layout(
            title=f"Sentiment Trends for {details['product_name']}",
            xaxis_title="Review Index",
            yaxis_title="Sentiment Scores",
            width=900,  # Set the desired width
            height=500,  # Set the desired height
            title_font=dict(size=20),  # Increase title font size
        )

        # Display each individual figure in Streamlit
        st.plotly_chart(fig)
# ________________________________________________________________________________________________________________________

elif (st.session_state.page == "Results") & (st.session_state.credentials_ok == False):
    st.title("Results")
    st.subheader("No results found!!")
