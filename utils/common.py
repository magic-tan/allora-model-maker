def snake_to_camel(snake_str):
    """Convert snake_case to CamelCase (PascalCase)."""
    components = snake_str.split("_")
    return "".join(x.capitalize() for x in components)
