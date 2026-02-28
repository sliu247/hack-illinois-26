from pdf_utils import process_pdf

chunks = process_pdf("/Users/25oveemuley/Desktop/HackIllinois/markov_chain.pdf")

print(f"Number of chunks: {len(chunks)}")
print("\nFirst chunk preview:\n")
print(chunks[0][:1000])