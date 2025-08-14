import json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from collections import defaultdict

# Chemins à adapter
model_id = "llava-hf/llava-1.5-7b-hf"
data_path = "C:/Users/sarra/Downloads/changechat/train/train/json/Train_50images.json"
image_folder = "C:/Users/sarra/Downloads/changechat/train/train/image"

def normalize_answer(answer):
    """Normalise la réponse pour la comparaison"""
    # Convertir en minuscules et supprimer les espaces en début/fin
    answer = answer.lower().strip()
    # Supprimer la ponctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Normaliser les espaces
    answer = ' '.join(answer.split())
    return answer

def extract_numerical_answer(answer):
    """Extrait les valeurs numériques de la réponse"""
    numbers = re.findall(r'\d+', answer)
    if numbers:
        return numbers[0]  # Retourne le premier nombre trouvé
    return answer

def categorize_answer(answer):
    """Catégorise la réponse selon les types de questions"""
    answer_lower = answer.lower()
    
    # Catégories pour les réponses de type ratio/percentage
    if any(word in answer_lower for word in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']):
        if any(word in answer_lower for word in ['0_to_10', '0-10', '0 to 10']):
            return '0_to_10'
        elif any(word in answer_lower for word in ['10_to_20', '10-20', '10 to 20']):
            return '10_to_20'
        elif any(word in answer_lower for word in ['20_to_30', '20-30', '20 to 30']):
            return '20_to_30'
        elif any(word in answer_lower for word in ['30_to_40', '30-40', '30 to 40']):
            return '30_to_40'
        elif any(word in answer_lower for word in ['40_to_50', '40-50', '40 to 50']):
            return '40_to_50'
        elif any(word in answer_lower for word in ['50_to_60', '50-60', '50 to 60']):
            return '50_to_60'
        elif any(word in answer_lower for word in ['60_to_70', '60-70', '60 to 70']):
            return '60_to_70'
        elif any(word in answer_lower for word in ['70_to_80', '70-80', '70 to 80']):
            return '70_to_80'
        elif any(word in answer_lower for word in ['80_to_90', '80-90', '80 to 90']):
            return '80_to_90'
        elif any(word in answer_lower for word in ['90_to_100', '90-100', '90 to 100']):
            return '90_to_100'
    
    # Catégories pour les réponses oui/non
    if any(word in answer_lower for word in ['yes', 'oui', 'true', 'vrai']):
        return 'yes'
    elif any(word in answer_lower for word in ['no', 'non', 'false', 'faux']):
        return 'no'
    
    # Catégories pour les types de changements
    if any(word in answer_lower for word in ['buildings', 'bâtiments', 'construction']):
        return 'buildings'
    elif any(word in answer_lower for word in ['water', 'eau', 'rivière', 'lac']):
        return 'water'
    elif any(word in answer_lower for word in ['vegetation', 'végétation', 'forêt', 'arbre']):
        return 'vegetation'
    
    return answer

def evaluate_llava():
    """Évalue le modèle LLaVA et calcule les métriques"""
    
    print("Chargement du modèle LLaVA...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Chargement des données...")
    with open(data_path, "r") as f:
        data = json.load(f)["question"]
    
    # Limiter à 10 exemples pour l'évaluation (pour éviter les timeouts)
    test_data = data[:10]
    
    print(f"Évaluation sur {len(test_data)} exemples...")
    
    y_true = []
    y_pred = []
    results = []
    
    for i, sample in enumerate(test_data):
        print(f"Traitement exemple {i+1}/{len(test_data)}...")
        
        img_id = sample["img_id"]
        im1_path = os.path.join(image_folder, "im1", img_id)
        question = sample["question"]
        expected_answer = sample["answer"]
        
        try:
            image = Image.open(im1_path).convert("RGB")
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
                generated_answer = processor.decode(output[0], skip_special_tokens=True)
            
            # Extraire seulement la réponse générée (après "ASSISTANT:")
            if "ASSISTANT:" in generated_answer:
                generated_answer = generated_answer.split("ASSISTANT:")[-1].strip()
            
            # Normaliser les réponses
            expected_normalized = normalize_answer(expected_answer)
            generated_normalized = normalize_answer(generated_answer)
            
            # Catégoriser les réponses
            expected_cat = categorize_answer(expected_normalized)
            generated_cat = categorize_answer(generated_normalized)
            
            y_true.append(expected_cat)
            y_pred.append(generated_cat)
            
            results.append({
                'question': question,
                'expected': expected_answer,
                'generated': generated_answer,
                'expected_cat': expected_cat,
                'generated_cat': generated_cat,
                'match': expected_cat == generated_cat
            })
            
            print(f"Q: {question}")
            print(f"Attendu: {expected_answer} -> {expected_cat}")
            print(f"Généré: {generated_answer} -> {generated_cat}")
            print(f"Match: {expected_cat == generated_cat}")
            print('-' * 40)
            
        except Exception as e:
            print(f"Erreur pour l'exemple {i+1}: {e}")
            continue
    
    # Calcul des métriques
    if len(y_true) > 0:
        # Accuracy simple
        accuracy = accuracy_score(y_true, y_pred)
        
        # Précision, rappel, F1 (macro average pour gérer les classes multiples)
        try:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except:
            # Si pas assez de classes, utiliser binary
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Compter les matches exacts
        exact_matches = sum(1 for r in results if r['match'])
        exact_match_rate = exact_matches / len(results) if results else 0
        
        print("\n" + "="*50)
        print("RÉSULTATS D'ÉVALUATION LLaVA")
        print("="*50)
        print(f"Nombre d'exemples évalués: {len(results)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Précision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Rappel: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"Exact Match Rate: {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
        print("="*50)
        
        # Détail des résultats
        print("\nDÉTAIL DES RÉSULTATS:")
        for i, result in enumerate(results):
            print(f"{i+1}. Q: {result['question'][:50]}...")
            print(f"   Attendu: {result['expected']}")
            print(f"   Généré: {result['generated']}")
            print(f"   Match: {'✓' if result['match'] else '✗'}")
            print()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match_rate': exact_match_rate,
            'results': results
        }
    else:
        print("Aucun exemple évalué avec succès.")
        return None

if __name__ == "__main__":
    results = evaluate_llava() 