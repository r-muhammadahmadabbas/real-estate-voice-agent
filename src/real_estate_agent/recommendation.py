"""
Property Recommendation Engine
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from real_estate_agent.config import PROPERTIES_PATH


@dataclass
class UserPreferences:
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    city: Optional[str] = None
    area: Optional[str] = None
    min_bedrooms: int = 0
    property_type: Optional[str] = None  # buy, rent, invest
    purpose: Optional[str] = None  # residential, commercial, plot
    amenities: Optional[List[str]] = None
    investment_goals: Optional[str] = None

class PropertyRecommendationEngine:
    def __init__(self, properties_path: str = None):
        self.properties_df = pd.read_csv(properties_path or PROPERTIES_PATH)
        self.properties_df['amenities'] = self.properties_df['amenities'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.properties_df['payment_plans'] = self.properties_df['payment_plans'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.properties_df['nearby_schools'] = self.properties_df['nearby_schools'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.properties_df['nearby_hospitals'] = self.properties_df['nearby_hospitals'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.properties_df['coordinates'] = self.properties_df['coordinates'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
    def filter_properties(self, prefs: UserPreferences) -> pd.DataFrame:
        """Filter properties by hard constraints"""
        df = self.properties_df[self.properties_df['status'] == 'available'].copy()
        
        if prefs.budget_max:
            df = df[df['price'] <= prefs.budget_max]
        if prefs.budget_min:
            df = df[df['price'] >= prefs.budget_min]
        if prefs.city:
            df = df[df['city'].str.lower() == prefs.city.lower()]
        if prefs.area:
            df = df[df['area'].str.lower().str.contains(prefs.area.lower())]
        if prefs.min_bedrooms:
            df = df[df['bedrooms'] >= prefs.min_bedrooms]
        if prefs.property_type:
            df = df[df['type'].str.lower() == prefs.property_type.lower()]
        if prefs.purpose:
            if prefs.purpose == 'residential':
                df = df[df['type'].isin(['house', 'plot'])]
            elif prefs.purpose == 'commercial':
                df = df[df['type'] == 'commercial']
                
        return df
    
    def calculate_score(self, row: pd.Series, prefs: UserPreferences) -> float:
        """Calculate recommendation score for a property"""
        score = 0.0
        
        # Area preference match
        if prefs.area and prefs.area.lower() in str(row['area']).lower():
            score += 30
        elif prefs.city and prefs.city.lower() in str(row['city']).lower():
            score += 15
            
        # Budget fit (closer to max budget = higher score)
        if prefs.budget_max and row['price'] > 0:
            budget_ratio = row['price'] / prefs.budget_max
            if 0.5 <= budget_ratio <= 1.0:
                score += 20 * (1 - abs(1 - budget_ratio))
            elif budget_ratio < 0.5:
                score += 10  # Under budget is good but not perfect
                
        # Bedroom match
        if prefs.min_bedrooms and row['bedrooms'] >= prefs.min_bedrooms:
            score += 10
            
        # Amenity match
        if prefs.amenities and row['amenities']:
            prop_amenities = set(a.lower() for a in row['amenities'])
            user_amenities = set(a.lower() for a in prefs.amenities)
            matches = len(prop_amenities & user_amenities)
            score += matches * 5
            
        # Investment scoring
        if prefs.purpose == 'invest' or prefs.investment_goals:
            roi_score = self._calculate_roi_score(row)
            score += roi_score * 0.5
            
        # Property type match
        if prefs.property_type and row['type'] == prefs.property_type:
            score += 10
            
        return score
    
    def _calculate_roi_score(self, row: pd.Series) -> float:
        """Calculate investment potential score"""
        score = 0
        
        # DHA areas typically appreciate well
        if 'DHA' in str(row['area']) or 'Bahria' in str(row['area']):
            score += 20
            
        # Commercial properties have higher yields
        if row['type'] == 'commercial':
            score += 15
            
        # Lower price per sqft = better value
        if row['price_per_sqft'] < 15000:
            score += 10
        elif row['price_per_sqft'] < 20000:
            score += 5
            
        # Payment plan flexibility
        plans = row['payment_plans']
        if plans and len(plans) > 0:
            plan = plans[0]
            if plan.get('installments', 0) >= 48:
                score += 5
                
        return score
    
    def recommend(self, prefs: UserPreferences, top_k: int = 5) -> List[Dict]:
        """Get top property recommendations"""
        candidates = self.filter_properties(prefs)
        
        if candidates.empty:
            return []
            
        # Score each candidate
        scored = []
        for _, row in candidates.iterrows():
            score = self.calculate_score(row, prefs)
            prop_dict = row.to_dict()
            prop_dict['recommendation_score'] = score
            scored.append(prop_dict)
            
        # Sort by score
        scored.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        # Format for response
        results = []
        for prop in scored[:top_k]:
            # Calculate monthly installment
            monthly = None
            if prop['payment_plans'] and len(prop['payment_plans']) > 0:
                plan = prop['payment_plans'][0]
                down = prop['price'] * plan.get('down', 0) / 100
                remaining = prop['price'] - down
                monthly = remaining / plan.get('installments', 1) if plan.get('installments') else None
                
            results.append({
                'id': prop['id'],
                'title': prop['title'],
                'type': prop['type'],
                'city': prop['city'],
                'area': prop['area'],
                'price': prop['price'],
                'price_formatted': f"PKR {prop['price']:,.0f}",
                'area_sqft': prop['area_sqft'],
                'bedrooms': prop['bedrooms'],
                'bathrooms': prop['bathrooms'],
                'amenities': prop['amenities'],
                'payment_plan': prop['payment_plans'][0] if prop['payment_plans'] else None,
                'monthly_installment': monthly,
                'score': prop['recommendation_score'],
                'possession_date': prop['possession_date'],
                'coordinates': prop['coordinates']
            })
            
        return results


# Budget parser for UrduLish
def parse_budget(text: str) -> Dict[str, Optional[int]]:
    """Parse budget from UrduLish text like '3 crore', '50 lakh', '2.5 crore'"""
    text = text.lower().replace(',', '')
    
    # Extract numbers with crore/lakh
    import re
    
    # Pattern: number + crore/lakh
    crore_match = re.search(r'(\d+(?:\.\d+)?)\s*crore', text)
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*lakh', text)
    
    budget_pkr = 0
    if crore_match:
        budget_pkr += float(crore_match.group(1)) * 10000000
    if lakh_match:
        budget_pkr += float(lakh_match.group(1)) * 100000
        
    # If just a number, assume crores if > 100, else lakhs
    if budget_pkr == 0:
        num_match = re.search(r'(\d+(?:\.\d+)?)', text)
        if num_match:
            num = float(num_match.group(1))
            if num > 100:
                budget_pkr = num * 10000000  # assume crores
            else:
                budget_pkr = num * 100000    # assume lakhs
                
    return {
        'budget_min': int(budget_pkr * 0.7) if budget_pkr > 0 else None,
        'budget_max': int(budget_pkr * 1.2) if budget_pkr > 0 else None,
        'exact': int(budget_pkr) if budget_pkr > 0 else None
    }


def parse_preferences_from_text(text: str) -> Dict:
    """Extract preferences from conversation text"""
    prefs = {
        'budget_min': None,
        'budget_max': None,
        'city': None,
        'area': None,
        'min_bedrooms': 0,
        'property_type': None,
        'purpose': None,
        'amenities': [],
        'investment_goals': None
    }
    
    text_lower = text.lower()
    
    # Budget
    budget = parse_budget(text)
    prefs['budget_min'] = budget['budget_min']
    prefs['budget_max'] = budget['budget_max']
    
    # City detection
    cities = ['lahore', 'islamabad', 'rawalpindi', 'karachi', 'multan', 'faisalabad']
    for city in cities:
        if city in text_lower:
            prefs['city'] = city.capitalize()
            break
            
    # Area detection
    areas = ['dha phase 5', 'dha phase 6', 'dha phase 8', 'dha phase 9', 'dha phase 9 prism',
             'bahria town phase 7', 'bahria town phase 8', 'bahria enclave',
             'gulberg iii', 'gulberg green', 'gulberg']
    for area in areas:
        if area in text_lower:
            prefs['area'] = area.title()
            break
            
    # Bedrooms
    import re
    bed_match = re.search(r'(\d+)\s*(?:bed|bedroom)', text_lower)
    if bed_match:
        prefs['min_bedrooms'] = int(bed_match.group(1))
        
    # Property type
    if any(w in text_lower for w in ['buy', 'purchase', 'apna ghar']):
        prefs['property_type'] = 'buy'
    elif any(w in text_lower for w in ['rent', 'kira']):
        prefs['property_type'] = 'rent'
    elif any(w in text_lower for w in ['invest', 'investment', 'returns']):
        prefs['property_type'] = 'invest'
        
    # Purpose
    if 'commercial' in text_lower or 'dukan' in text_lower or 'office' in text_lower:
        prefs['purpose'] = 'commercial'
    elif 'plot' in text_lower or 'zameen' in text_lower:
        prefs['purpose'] = 'plot'
    elif 'house' in text_lower or 'ghar' in text_lower:
        prefs['purpose'] = 'residential'
        
    # Amenities
    amenity_keywords = {
        'park': 'park',
        'gym': 'gym',
        'pool': 'pool',
        'swimming': 'pool',
        'mosque': 'mosque',
        'masjid': 'mosque',
        'security': 'security',
        'guard': 'security',
        'community center': 'community_center',
        'market': 'market',
        'wide road': 'wide_roads'
    }
    for kw, amenity in amenity_keywords.items():
        if kw in text_lower:
            prefs['amenities'].append(amenity)
            
    # Investment goals
    if any(w in text_lower for w in ['invest', 'return', 'yield', 'profit', 'appreciation']):
        prefs['investment_goals'] = 'capital_appreciation'
        
    return prefs


if __name__ == "__main__":
    # Test the recommendation engine
    engine = PropertyRecommendationEngine()
    
    # Test preferences
    prefs = UserPreferences(
        budget_max=30000000,
        city="Lahore",
        area="DHA",
        min_bedrooms=0,
        property_type="buy",
        purpose="residential"
    )
    
    results = engine.recommend(prefs, top_k=5)
    print(f"Found {len(results)} recommendations:")
    for r in results:
        print(f"  {r['title']} - {r['price_formatted']} - Score: {r['score']:.1f}")
        if r['monthly_installment']:
            print(f"    Monthly: PKR {r['monthly_installment']:,.0f}")