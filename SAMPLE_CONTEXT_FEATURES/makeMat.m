% place the features filename and 
% synonyms filename (or hyponyms etc) in their correct place.
% F.features = 'es.ortho.v1';
% F.syns = 'es.syns.v1';
% 
% F.features = 'en.ortho.v1';
% F.syns = 'en.syns.v1';
% 
% 
% F.features = 'FEB3_en.features.10k';
% F.syns = 'en.syns.v1';
% 
% F.features = 'FEB3_es.features.10k';
% F.syns = 'es.syns.v1';

% F.features = 'FEB6_en.features_space.10k';
% F.syns = 'en.syns.v1';
% lexsyn2mat(F);
%F.features = 'FEB6_en.features_space.10k';
%F.syns = 'en.syns.v2';

F.features = 'context_features.en.2';
F.syns = '';
lexsyn2mat(F);