-- Define the base path for reuse
DEFINE base_path='/Users/chantimyeung/Dropbox/Mac/Desktop/Codes/';

-- Execute scripts in the desired order
@&base_path.create_tables.ddl;
@&base_path.sequences.sql;
@&base_path.functions.sql;
@&base_path.procedures.sql;
@&base_path.package.sql;
@&base_path.triggers.sql;
@&base_path.insert_data.sql;
@&base_path.scheduled_jobs.sql;
@&base_path.views.sql;
@&base_path.alternate_index.sql;
@&base_path.roles.sql;
