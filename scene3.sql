-- Test Procedure: Filter Options By Category
BEGIN
    FilterOptionsByCategory(NULL);  -- For 'Cty' category
END;
/

BEGIN
    FilterOptionsByCategory('Electrical');
END;
/
