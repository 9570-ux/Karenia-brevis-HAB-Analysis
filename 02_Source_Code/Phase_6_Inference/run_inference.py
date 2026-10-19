"""Student CLI for a single recorded observation; no files are written."""
import argparse
import json
from inference import FIELDS, OPTIONAL, predict_observation


def main():
    parser = argparse.ArgumentParser(description='Predicted recorded abundance category using the saved Phase 5 model. Not a future HAB forecast.')
    for field in FIELDS:
        parser.add_argument('--' + field.replace('_', '-'), type=float,
                            required=field not in OPTIONAL,
                            help='Missing: omit this option; saved imputer handles it.' if field in OPTIONAL else 'Required recorded observation value.')
    parser.add_argument('--json', action='store_true', help='Print the complete reusable API result as JSON.')
    args = vars(parser.parse_args())
    as_json = args.pop('json')
    try:
        result = predict_observation(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    if as_json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print('Prediction Result\n-----------------')
        print(f"{result['terminology']}: {result['predicted_class']}\n\nClass probabilities:")
        for label, probability in result['class_probabilities'].items():
            print(f'{label}: {probability:.6f}')
        print('\nInput validation: PASS')
        print('Missing environmental fields: ' + (', '.join(result['input_validation']['missing_environmental_fields']) or 'none'))
        print('Model: ' + result['model']['identity'])
        print('Model artifact: ' + result['model']['artifact'])
        print(result['scientific_limitation'])


if __name__ == '__main__':
    main()
