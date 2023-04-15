/// Splits document into a vector of words
pub fn tokenize(document: &str) -> Vec<String> {
	document
		.to_lowercase()
		.split(|c: char| !c.is_alphabetic())
		.filter(|s: &&str| !s.is_empty())
		.map(|s: &str| s.to_string())
		.collect()
}
