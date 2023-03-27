import streamlit as st

# Set page title
st.set_page_config(page_title="Emoture - An Emotion Recognition App")

# Create a container to hold the navigation bar
nav_container = st.container()

# Add a row to the container with two columns
cols = nav_container.columns([2, 10])

# Add content to the left column
with cols[0]:
    st.image('C:/Users/Rahul Maddula/PycharmProjects/OCH/images/OCH_Logo.png', width=50)

# Add content to the right column
with cols[1]:
    st.title("My Awesome Streamlit App")

    # Add a horizontal line for separation
    st.write("---")

    # Add links to the navigation bar
    st.write("[Home](#)")
    st.write("[About](#)")
    st.write("[Contact](#)")

# Add a banner image
st.image("homepageImage.jpeg", use_column_width=True, caption="Image Source: Unsplash")

# Add some content sections
st.markdown("<h2 id='about'>About</h2>", unsafe_allow_html=True)
st.write("""
    This app is a simple user interface for our IR Project - An Integrated Approach to Context-aware Emotion Recognition through Posture and Speech Modulation Analysis. 
    Through our ML models, one's emotional state can be inferred by analysing their posture and speech modulation, while also being contextually aware.
""")

st.markdown("<h2 id='upload'>Upload File</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an MP4 video file to upload.", type="mp4")
if uploaded_file is not None:
    st.video(uploaded_file)

st.markdown("<h2 id='contact'>Contact</h2>", unsafe_allow_html=True)
st.write("""
    If you have any questions or feedback about the app, please send an email to example@example.com.
""")


# Set up navigation bar
st.write("""
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .stApp header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: #333333;
            color: #ffffff;
            padding: 0.5rem;
        }
        .stApp header a {
            color: #ffffff;
            text-decoration: none;
            margin-right: 1rem;
        }
        .stApp header a:hover {
            text-decoration: underline;
        }
        .content-section {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .content-box {
            background-color: #ffffff;
            border: 1px solid #e1e1e1;
            border-radius: 5px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            width: calc(33.33% - 1rem);
        }
        .content-box img {
            width: 100%;
            margin-bottom: 1rem;
        }
        .content-box h3 {
            margin-top: 0;
        }
    </style>
""", unsafe_allow_html=True)
st.write("<header><a href='#'>Home</a><a href='#'>Upload Video</a></header>", unsafe_allow_html=True)

# Set up sidebar menu
st.sidebar.title("Menu")
menu_options = ["Home", "Upload Video"]
menu_choice = st.sidebar.selectbox("Select an option", menu_options)

# Display appropriate page based on menu choice
if menu_choice == "Home":
    # Add a banner image
    st.image("homepageImage.jpeg", use_column_width=True)

    # Add some content sections
    st.markdown("## What is this app?")
    st.write("""
        This app is a simple user interface for our IR Project - An Integrated Approach to Context-aware Emotion Recognition through Posture and Speech Modulation Analysis. 
        Through our ML models, one's emotional state can be inferred by analysing their posture and speech modulation, while also being contextually aware.
    """)

    st.markdown("## Why use this app?")
    st.write("""
        There are several benefits to using this app to play your videos:
        - No need to download any software
        - Works on any device with a web browser
        - Plug and play, easy and quick to use.
    """)

    st.markdown("## How to use this app?")
    st.write("""
        1. Select 'Upload Video' from the menu.
        2. Choose an MP4 video file to upload.
        3. Wait for the video to upload and process (this may take a while for large files).
        4. Wait for the processing and view your results!
        """)

    # Add some featured content boxes
    st.markdown("## Featured Videos")
    st.write("<div class='content-section'>", unsafe_allow_html=True)

    st.write("""
        <div class='content-box'>
            <img src='https://www.google.com/url?sa=i&url=https%3A%2F%2Fthenextweb.com%2Fnews%2Femotion-recognition-ai-cant-determine-how-people-feel&psig=AOvVaw2K9FFgSV-U0qzspBYvhtPj&ust=1679770357870000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCLC5ia2e9f0CFQAAAAAdAAAAABAD' alt='Video 1'>
            <h3>Video 1</h3>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
                <a href='#'>Watch Now</a>
            </div>
        """, unsafe_allow_html=True)

    st.write("""
            <div class='content-box'>
                <img src='https://images.unsplash.com/photo-1541743863329-0ca247c8a9cf' alt='Video 2'>
                <h3>Video 2</h3>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam.</p>
                <a href='#'>Watch Now</a>
            </div>
        """, unsafe_allow_html=True)

    st.write("""
            <div class='content-box'>
                <img src='https://images.unsplash.com/photo-1536811051087-4dbbe968c19d' alt='Video 3'>
                <h3>Video 3</h3>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam.</p>
                <a href='#'>Watch Now</a>
            </div>
        """, unsafe_allow_html=True)

    st.write("</div>", unsafe_allow_html=True)

else:
    # Add upload form
    st.markdown("## Upload a video")
    uploaded_file = st.file_uploader("Choose an MP4 video", type="mp4")

    # Perform emotion recognition analysis on the uploaded video and display the results
    if uploaded_file:
        result_container = st.empty()

        # Show a loading message while we process the video
        with st.spinner('Processing the video...'):
            # Perform emotion recognition analysis here and get the results
            # Add code here to perform the analysis
            results = {
                'happy': 0.2,
                'sad': 0.3,
                'angry': 0.4,
                'neutral': 0.1
            }

        # Display the results
        result_container.markdown("## Emotion Recognition Results")
        for emotion, score in results.items():
            result_container.write(f"- **{emotion.title()}**: {score:.2f}")


# Below code is to hide default hamburger menu and streamlit footer

# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
