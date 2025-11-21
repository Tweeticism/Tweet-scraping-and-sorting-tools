Hydra 4.0.0
________________________________________
Overview
•	Tweet hydrator, for hydrating tweet IDs without Twitter API. Uses Playwright.
•	Prioritizes hydration rate over accuracy

Features
•	High hydration rate, at the risk of clutter and false positives
•	Supports emojis, and detects media and links
•	Parallel processing for speed (8 workers default)
•	Real-time saving/writing and robust resume logic 
•	No Twitter API required
•	Progress bars
•	Headless mode
•	Dynamic file name detection
•	Only processes lists of tweet IDs saved in .txt format

Ethical use 
•	Only hydrates public tweets and public information—ignores protected/private, age-restricted, and deleted tweets, in compliance with data protection/ethics regimes
•	Throttle function, to comply with Twitter/X Terms of Service

Usage
•	Put Hydra in the same folder as the input files
•	Input files should have “tweet_id” in the filename
•	Run Hydra 

Scylla 4.3.0
________________________________________
Overview
•	Ingests a large number of tweets, and then cleans, sorts and de-duplicates them 
•	Designed to sort large batches of tweets via keywords

Features
•	Uses a set of keywords (see attached humanitarian_keywords_and_categories.txt file) to sort a large number of tweets into five humanitarian categories:	red_cross; un_agencies; msf_solidarist;	faith_based; and others (general mentions of humanitarian organizations)
•	Normalizes character sets, emojis and special characters
•	Uses tweet IDs to identify and remove duplicates
•	Processes only .csv files

Usage
•	Create a folder, put Scylla in it
•	Create a sub-named “tweet_data” and put all input .csv files in it
•	Run Scylla

Cassandra 3.1.0
________________________________________
Overview
•	Ingests a number of tweets and returns an analysis of shared narratives
•	Uses BERTopic to process the tweets and to model shared topics/narratives

Features
•	Finds common narratives shared by at least 15 tweets
•	Returns a number of analytical files (see folders), including visualizations, sentiment analysis, and timeline analysis

Usage
•	Set path of input file in code
•	Run Cassandra

Licenses
________________________________________
•	Standard MIT Licenses

Attributions and Acknowledgments
________________________________________
•	These projects were developed with the assistance of Microsoft Copilot. The author designed the system architecture, implemented core logic, and performed all decisions, validations, and interpretations—he only used AI to assist with debugging, refining and drafting code.
•	BERTopic (Grootendorst, 2022) was used to power Cassandra.
•	Microsoft Playwright was used to power Hydra.


Citations
________________________________________
Microsoft. (2025). Copilot [AI assistant]. Microsoft. https://copilot.microsoft.com/
Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
Microsoft. (2020). Playwright [Computer software]. https://playwright.dev/

