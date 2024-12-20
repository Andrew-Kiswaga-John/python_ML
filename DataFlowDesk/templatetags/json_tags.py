from django import template
import json
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='json_encode')
def json_encode(value):
    """Safely encode a value as JSON."""
    return mark_safe(json.dumps(value))
