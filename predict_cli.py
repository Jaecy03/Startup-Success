import argparse
import os
from main import predict_startup_success
from visualize import visualize_prediction, visualize_model_comparison

def main():
    parser = argparse.ArgumentParser(description="Predict startup success probability")

    # Create argument groups
    prediction_args = parser.add_argument_group('prediction arguments')
    visualization_args = parser.add_argument_group('visualization arguments')

    # Prediction arguments
    prediction_args.add_argument("--name", help="Startup name")
    prediction_args.add_argument("--funding", type=float, help="Total funding in USD")
    prediction_args.add_argument("--rounds", type=int, help="Number of funding rounds")
    prediction_args.add_argument("--category", help="Business category")

    # Visualization arguments
    visualization_args.add_argument("--visualize", action="store_true", help="Show visualization of prediction")
    visualization_args.add_argument("--save-viz", help="Save visualization to specified path")
    visualization_args.add_argument("--compare-models", action="store_true", help="Show model comparison visualization")

    args = parser.parse_args()

    try:
        # Show model comparison if requested
        if args.compare_models:
            save_path = None
            if args.save_viz:
                save_path = os.path.join(
                    os.path.dirname(args.save_viz) if os.path.dirname(args.save_viz) else "visualizations",
                    "model_comparison.png"
                )
            visualize_model_comparison(save_path=save_path)
            return

        # Validate required arguments for prediction
        if not all([args.name, args.funding is not None, args.rounds is not None, args.category]):
            parser.error("--name, --funding, --rounds, and --category are required for prediction")

        # Get prediction
        prediction = predict_startup_success(
            startup_name=args.name,
            funding_total_usd=args.funding,
            funding_rounds=args.rounds,
            category=args.category
        )

        # Print text results
        print("\n🔮 Startup Prediction:")
        print(f"Startup: {prediction['startup_name']}")
        print(f"Success Probability: {prediction['success_probability']}%")
        print(f"Prediction: {prediction['prediction']}")

        # Visualize if requested
        if args.visualize or args.save_viz:
            save_path = args.save_viz
            if args.visualize and not save_path:
                save_path = f"visualizations/{args.name.replace(' ', '_')}_prediction.png"

            visualize_prediction(prediction, save_path=save_path)
            print(f"\n✅ Visualization saved to {save_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()