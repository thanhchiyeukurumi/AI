import tkinter as tk
from tkinter import ttk, scrolledtext
from prediction.predict import VotingSpamDetector

class EnhancedSpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize spam detector
        self.spam_detector = VotingSpamDetector()
        
        self.setup_single_test_interface()
    
    def setup_single_test_interface(self):
        # Title
        title_label = ttk.Label(
            self.root,
            text="Kiểm tra tin nhắn spam",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Message input
        input_frame = ttk.LabelFrame(self.root, text="Nhập tin nhắn", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.message_input = scrolledtext.ScrolledText(
            input_frame, 
            height=10, 
            font=("Helvetica", 11)
        )
        self.message_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Check button
        self.check_button = ttk.Button(
            btn_frame,
            text="Kiểm tra Spam",
            command=self.check_spam
        )
        self.check_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(
            btn_frame,
            text="Xóa tin nhắn",
            command=lambda: self.message_input.delete("1.0", tk.END)
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(self.root, text="Kết quả", padding="10")
        result_frame.pack(fill=tk.X, pady=10, padx=10)
        
        self.result_label = ttk.Label(
            result_frame,
            text="",
            font=("Helvetica", 12)
        )
        self.result_label.pack(pady=5)
    
    def check_spam(self):
        message = self.message_input.get("1.0", tk.END).strip()
        if message:
            # Get weighted spam score from VotingSpamDetector
            try:
                is_spam = self.spam_detector.is_spam(message)
                
                # Update UI
                result_text = "TIN NHẮN LÀ SPAM!" if is_spam else "Tin nhắn KHÔNG PHẢI là spam."
                self.result_label.configure(
                    text=result_text,
                    foreground="red" if is_spam else "green"
                )
                
            except Exception as e:
                self.result_label.configure(
                    text=f"Lỗi: {str(e)}",
                    foreground="black"
                )
        else:
            self.result_label.configure(
                text="Vui lòng nhập tin nhắn để kiểm tra",
                foreground="black"
            )

def main():
    root = tk.Tk()
    app = EnhancedSpamDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 