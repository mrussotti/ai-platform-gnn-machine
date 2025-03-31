import re
import datetime

def are_same_incident(incident1, incident2, threshold=0.75):
    """
    Determines if two incident representations refer to the same real-world emergency event.
    
    Args:
        incident1: Dict containing first incident data (from Neo4j or API response)
        incident2: Dict containing second incident data (from Neo4j or API response)
        threshold: Minimum confidence score to consider incidents as matching
        
    Returns:
        Dict containing:
        - is_duplicate: Boolean indicating if incidents are the same
        - confidence_score: Overall confidence score (0-1)
        - match_details: Dict with individual similarity scores
    """
    # Initialize scores and weights
    match_details = {}
    weights = {
        "location": 0.4,  # Location is highest weight
        "time": 0.3,      # Time is second highest
        "nature": 0.2,    # Incident type is third
        "details": 0.1    # Other details are lowest weight
    }
    confidence_score = 0.0
    
    # 1. LOCATION MATCHING
    # Get addresses to compare
    addr1 = incident1.get("location", {}).get("address", "")
    addr2 = incident2.get("location", {}).get("address", "")
    
    location_score = 0.0
    if addr1 and addr2:
        # Simple address matching (can be made more sophisticated)
        addr1 = addr1.lower().strip()
        addr2 = addr2.lower().strip()
        
        # Extract any numbers (house numbers, lot numbers)
        nums1 = set(re.findall(r'\d+', addr1))
        nums2 = set(re.findall(r'\d+', addr2))
        
        # Check for exact matches
        if addr1 == addr2:
            location_score = 1.0
        # Check for significant overlap
        elif addr1 in addr2 or addr2 in addr1:
            location_score = 0.8
        # Check for number matches (street numbers, lot numbers)
        elif nums1 and nums2 and nums1.intersection(nums2):
            location_score = 0.7
        # Check for partial matches (might be same street)
        elif any(word in addr2.split() for word in addr1.split() if len(word) > 3):
            location_score = 0.5
    
    match_details["location_score"] = location_score
    confidence_score += location_score * weights["location"]
    
    # 2. TIME MATCHING
    # Get timestamps to compare
    time1 = incident1.get("location", {}).get("time", "")
    time2 = incident2.get("location", {}).get("time", "")
    
    time_score = 0.0
    if time1 and time2:
        try:
            # Convert to datetime if strings
            if isinstance(time1, str):
                time1 = datetime.datetime.fromisoformat(time1.replace('Z', '+00:00'))
            if isinstance(time2, str):
                time2 = datetime.datetime.fromisoformat(time2.replace('Z', '+00:00'))
                
            # Calculate time difference in minutes
            time_diff = abs((time1 - time2).total_seconds() / 60)
            
            # Score based on time proximity (within 30 min window)
            if time_diff < 5:
                time_score = 1.0
            elif time_diff < 15:
                time_score = 0.8
            elif time_diff < 30:
                time_score = 0.5
            else:
                time_score = 0.0
        except:
            # If time parsing fails, neutral score
            time_score = 0.5
    
    match_details["time_score"] = time_score
    confidence_score += time_score * weights["time"]
    
    # 3. NATURE/TYPE MATCHING
    # Get incident types to compare
    nature1 = incident1.get("incident", {}).get("nature", "")
    nature2 = incident2.get("incident", {}).get("nature", "")
    
    nature_score = 0.0
    if nature1 and nature2:
        nature1 = nature1.lower().strip()
        nature2 = nature2.lower().strip()
        
        # Check for exact matches
        if nature1 == nature2:
            nature_score = 1.0
        # Check for partial matches
        elif nature1 in nature2 or nature2 in nature1:
            nature_score = 0.8
        # Check for related emergency types
        else:
            # Define related emergency types
            related_types = [
                ["fire", "smoke", "burning", "flames"],
                ["medical", "heart", "breathing", "unconscious"],
                ["accident", "crash", "collision", "vehicle"],
                ["theft", "robbery", "burglary", "stolen"]
            ]
            
            # Check if both natures belong to the same group
            for group in related_types:
                if any(term in nature1 for term in group) and any(term in nature2 for term in group):
                    nature_score = 0.7
                    break
    
    match_details["nature_score"] = nature_score
    confidence_score += nature_score * weights["nature"]
    
    # 4. DETAILS MATCHING
    # Get additional details to compare
    summary1 = incident1.get("call", {}).get("summary", "")
    summary2 = incident2.get("call", {}).get("summary", "")
    
    details_score = 0.0
    if summary1 and summary2:
        summary1 = summary1.lower()
        summary2 = summary2.lower()
        
        # Simple overlap of key terms
        words1 = set(summary1.split())
        words2 = set(summary2.split())
        
        # Calculate Jaccard similarity (intersection over union)
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            details_score = overlap / total if total > 0 else 0
    
    match_details["details_score"] = details_score
    confidence_score += details_score * weights["details"]
    
    # Final decision
    is_duplicate = confidence_score >= threshold
    
    return {
        "is_duplicate": is_duplicate,
        "confidence_score": confidence_score,
        "match_details": match_details
    }