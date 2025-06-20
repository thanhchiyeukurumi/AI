from prediction.predict import VotingSpamDetector

# Tạo đối tượng detector
spam_detector = VotingSpamDetector()

# Dự đoán tin nhắn
message = "good morning"
is_spam = spam_detector.is_spam(message)
print(f"Tin nhắn: {message}")
print(f"Is spam: {is_spam}")