import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Improved Chatbot with Enhanced Neural Network')
    parser.add_argument('--mode', choices=['train', 'experiment', 'test', 'chat'], default='chat',
                        help='Mode to run the chatbot (train, experiment, test, chat)')
    parser.add_argument('--model', choices=['original', 'enhanced'], default='enhanced',
                        help='Which model to use (original or enhanced)')

    args = parser.parse_args()

    if args.mode == 'train':
        from train import train_chatbot
        if args.model == 'original':
            print("Training original model...")
            train_chatbot()
        else:
            print("Training enhanced model...")
            train_chatbot('intents.json', 'enhanced_model_weights.pth', 'enhanced_model_data.json')

    elif args.mode == 'experiment':
        from experiment_model_types import experiment_architectures
        print("Running experiments to compare different architectures...")
        experiment_architectures()

    elif args.mode == 'test':
        from test_model import test_model
        if args.model == 'original':
            print("Testing original model...")
            test_model()
        else:
            print("Testing enhanced model...")
            if os.path.exists('enhanced_model_weights.pth'):
                test_model('enhanced_model_weights.pth', 'enhanced_model_data.json')
            else:
                print("Enhanced model not found. Please train it first with --mode train --model enhanced")

    elif args.mode == 'chat':
        from chat import SimpleChatBot
        print(f"Starting chat with {args.model} model...")
        if args.model == 'original':
            chatbot = SimpleChatBot()
        else:
            if os.path.exists('enhanced_model_weights.pth'):
                chatbot = SimpleChatBot('enhanced_model_weights.pth', 'enhanced_model_data.json')
            else:
                print("Enhanced model not found. Using original model instead.")
                chatbot = SimpleChatBot()

        bot_name = "TravelBot"
        print(f"{bot_name}: Hello! I'm your travel assistant. I can help you with:")
        print("• Landmarks and attractions in Paris, London, and Tokyo")
        print("• Local food recommendations and dining tips")
        print("• Weather information and best times to visit")
        print("• Cultural experiences and local customs")
        print("\nHow can I help you plan your journey? (Type 'quit' to exit)")

        while True:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                break

            response = chatbot.get_response(sentence)
            print(f"{bot_name}: {response}")


if __name__ == '__main__':
    main()