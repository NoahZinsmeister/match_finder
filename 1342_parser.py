import os
import re
import nltk
import csv

if __name__ == "__main__":
    book = []
    with open(os.getcwd() + "/1342.txt", "r") as readfile:
        book = readfile.read()    
    # removes non-book introductory and closing text
    book = book.split("START OF THIS PROJECT GUTENBERG EBOOK")[1]
    book = book.split("END OF THIS PROJECT GUTENBERG EBOOK")[0]
    # breaks book up into chapters
    chapters = re.split("\n\nChapter\s*[0-9]+\s*\n\n", book)[1:]
    # writes each chapter to a file in cwd/chapters
    for i, chapter in enumerate(chapters):
        with open(os.getcwd() + "/1342_chapters/chapter_" + str(i+1) +
                  ".csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            paragraphs = re.split("\n\n+", chapter)
            for paragraph in paragraphs:
                # concatenates fixed-length lines
                paragraph = paragraph.replace("\n", " ")
                # uses nltk to find sentences
                sentences = nltk.tokenize.sent_tokenize(paragraph)
                # writes each sentence to the active file
                for sentence in sentences:
                    writer.writerow([sentence])