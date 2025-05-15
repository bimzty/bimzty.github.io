-- ========================== Alternate Index: Options List Category ==========================

CREATE INDEX idx_option_category ON option_list(OPTION_CATEGORY);

SELECT INDEX_NAME, TABLE_NAME, STATUS
FROM USER_INDEXES
WHERE INDEX_NAME = 'IDX_OPTION_CATEGORY';
-- ========================== Alternate Index: Options List Category ==========================
