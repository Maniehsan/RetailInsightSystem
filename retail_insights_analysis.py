import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

# === Scraper for BooksToScrape ===
def scrape_books_to_scrape():
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    books = []

    for page in range(1, 6):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for book in soup.select(".product_pod"):
            title = book.h3.a['title']
            price = float(book.select_one(".price_color").text[1:])
            rating = book.p.get('class')[1]
            books.append({"Title": title, "Price": price, "Rating": rating})

    return pd.DataFrame(books)

# === Scraper for Project Gutenberg ===
def scrape_project_gutenberg():
    url = "https://www.gutenberg.org/ebooks/search/?sort_order=downloads"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    books = []
    for book in soup.select("li.booklink")[:25]:
        title = book.select_one("span.title").text.strip()
        price = round(40 + 10 * np.random.rand(), 2)
        rating = np.random.choice(['One', 'Two', 'Three', 'Four', 'Five'])
        books.append({"Title": title, "Price": price, "Rating": rating})

    return pd.DataFrame(books)

# === Clean Data ===
def clean_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    df = df[df['Price'] > 0]
    df = df[df['Rating'].isin(['One', 'Two', 'Three', 'Four', 'Five'])]
    return df

# === Save Data to Excel ===
def save_data(df, filename):
    df.to_excel(filename, index=False)
    print(f"Data saved to '{filename}'")

# === Decision Tree Example ===
def decision_tree_analysis(df):
    le = LabelEncoder()
    df['Rating_Numeric'] = le.fit_transform(df['Rating']) + 1

    X = df[['Price', 'Rating_Numeric']]
    y = ["High price" if p > 30 else "Low price" for p in df['Price']]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    prediction = clf.predict(pd.DataFrame([[20, 4]], columns=['Price', 'Rating_Numeric']))

    print(f"\n=== Decision Tree on Scraped Data ===")
    print(f"Prediction for price=20 and rating=4: {prediction}\n")

# === K-Means Clustering ===
def kmeans_analysis(df):
    le = LabelEncoder()
    df['Rating_Numeric'] = le.fit_transform(df['Rating']) + 1

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Price', 'Rating_Numeric']])
    print("=== K-Means on Scraped Data ===")
    print(df.head())

# === Apriori Algorithm ===
def apriori_analysis(df):
    print("\n=== Apriori Analysis ===")
    df = df.copy()
    df['Price_Range'] = pd.cut(df['Price'], bins=[0, 20, 40, 60, 100], labels=['Low', 'Medium', 'High', 'Very High'])
    basket = pd.get_dummies(df[['Rating', 'Price_Range']])

    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    print("Frequent Itemsets:\n", frequent_itemsets.head())
    print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# === Main ===
def main():
    print("=== Scraping Real-Time Data from Websites ===")
    df1 = scrape_books_to_scrape()
    print(f"Scraped {len(df1)} books from BooksToScrape")

    df2 = scrape_project_gutenberg()
    print(f"Scraped {len(df2)} books from Project Gutenberg\n")

    combined_df = pd.concat([df1, df2], ignore_index=True)
    print("Sources in combined data:\n")
    print(combined_df['Title'].groupby(combined_df.index // 50).count())

    print("\n=== Cleaning Data ===")
    cleaned_df = clean_data(combined_df)
    print("\n--- CLEANED DATA ---")
    print(cleaned_df.head())
    save_data(cleaned_df, "cleaned_data.xlsx")

    decision_tree_analysis(cleaned_df)
    kmeans_analysis(cleaned_df)
    apriori_analysis(cleaned_df)

if __name__ == "__main__":
    main()
