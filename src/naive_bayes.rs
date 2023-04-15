use std::collections::HashMap;
use crate::utils::tokenize;

/// Naive Bayes Classifier implementation
#[derive(Default)]
pub struct NaiveBayes {
	/// Frequency of each token in each class
	freq: HashMap<String, HashMap<String, usize>>,
	/// Frequency of each class
	class_freq: HashMap<String, usize>,
	/// Total documents count
	docs: usize,
}

impl NaiveBayes {
	pub fn new() -> NaiveBayes {
		NaiveBayes {
			freq: HashMap::new(),
			class_freq: HashMap::new(),
			docs: 0,
		}
	}

	/// Train the classifier on a given data
	pub fn train(&mut self, document: &str, class: &str) {
		// Turn document into a vector of tokens (words)
		let tokens = tokenize(document);

		for token in tokens {
			// Get a reference to a given class's frequency
			let class_freq = self.class_freq.entry(class.to_string()).or_default();
			// And increase it
			*class_freq += 1;

			// Get a reference to a token's frequency in each class
			let token_freq = self.freq.entry(token.to_string()).or_default();

			// Get a reference to a token's frequency in a given class
			let freq = token_freq.entry(class.to_string()).or_default();
			// And increase it
			*freq += 1;
		}

		self.docs += 1;
	}

	/// Predict class of the given document
	/// Returns None if the classifier was not trained
	pub fn predict(&self, document: &str) -> Option<String> {
		// Turn document into a vector of tokens (words)
		let tokens = tokenize(document);
		// This variable contains score of each class
		let mut scores = HashMap::new();

		for (class, class_freq) in &self.class_freq {
			// Set score based on the current class's
			// percentage in training data
			// .ln() here is a natural logarithm
			let mut score = (*class_freq as f64 / self.docs as f64).ln();

			for token in tokens.iter() {
				// Get frequency of a token in each class
				if let Some(class_freqs) = self.freq.get(token) {
					// Get frequency of a token in current class
					if let Some(token_freq) = class_freqs.get(class) {
						// Update score based on a token's average
						// frequency in the current class's documents
						// .ln() here is a natural logarithm
						score += (*token_freq as f64 / *class_freq as f64).ln();
					}
				}
			}

			scores.insert(class, score);
		}

		scores
			.into_iter()
			// Get the most possible class for the given document
			.min_by(|a, b| f64::partial_cmp(&a.1, &b.1).unwrap())
			// Turn class &str into String
			.map(|x| x.0.to_string())
	}
}

#[cfg(test)]
mod tests {
	use super::NaiveBayes;

	/// Tests Naive Bayes prediction implementation
	#[test]
	fn test_prediction() {
		let mut nb = NaiveBayes::new();

		let data = [
			("rust",       "fn use struct impl"),
			("python",     "def import from as"),
			("c",          "void include define ifdef"),
			("java",       "public static void main class"),
			("javascript", "let const function arrow"),
			("go",         "package import func var"),
			("swift",      "func var let struct class"),
			("kotlin",     "fun var val if else"),
			("typescript", "interface type import as"),
			("php",        "function include require echo"),
		];

		for (label, document) in data {
			nb.train(document, label);
		}

		for (label, document) in data {
			assert_eq!(label, nb.predict(document).unwrap());
		}
	}
}
