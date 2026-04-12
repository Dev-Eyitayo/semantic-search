"""Nigerian phone number normalization utilities."""
import re


def normalize_nigerian_phone(phone: str) -> str:
    """
    Normalize Nigerian phone numbers to +234xxxxx format.
    
    Accepts formats:
    - 08012345678 (11 digits starting with 0)
    - 2348012345678 (12 digits starting with 234)
    - +2348012345678 (with +234 prefix)
    
    Args:
        phone: Phone number string in various formats
        
    Returns:
        Normalized phone number in +234xxxxx format
        
    Raises:
        ValueError: If phone number format is invalid
    """
    if not phone:
        return None
        
    # Remove all whitespace, hyphens, and parentheses
    phone = re.sub(r'[\s\-\(\)]+', '', phone.strip())
    
    # Check if starts with +234
    if phone.startswith('+234'):
        # Already in correct format, just validate
        phone_digits = phone[4:]  # Get everything after +234
        if not re.match(r'^\d{10}$', phone_digits):
            raise ValueError("Invalid Nigerian phone number format. After country code should have 10 digits.")
        return phone
    
    # Check if starts with 234 (without +)
    elif phone.startswith('234'):
        phone_digits = phone[3:]  # Get everything after 234
        if not re.match(r'^\d{10}$', phone_digits):
            raise ValueError("Invalid Nigerian phone number format. After country code should have 10 digits.")
        return f"+{phone}"
    
    # Check if starts with 0 (Nigerian domestic format)
    elif phone.startswith('0'):
        if not re.match(r'^0\d{10}$', phone):
            raise ValueError("Invalid Nigerian phone number format. Should have 11 digits starting with 0.")
        # Convert 0XXXXXXXXXX to +234XXXXXXXXXX
        return f"+234{phone[1:]}"
    
    else:
        raise ValueError("Invalid Nigerian phone number format. Should start with 0, 234, or +234.")


def validate_nigerian_phone(phone: str) -> bool:
    """
    Validate Nigerian phone number format.
    
    Args:
        phone: Phone number string
        
    Returns:
        True if valid, False otherwise
    """
    if not phone:
        return False
        
    try:
        normalize_nigerian_phone(phone)
        return True
    except ValueError:
        return False
