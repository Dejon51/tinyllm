#include "cache.h"
#include <string.h>

void cache_init(ForwardCache *cache)
{
    memset(cache, 0, sizeof(ForwardCache));
    cache->length = 0;
}

void transformer_cache_init(TransformerBlockCache *cache)
{
    memset(cache, 0, sizeof(TransformerBlockCache));
    cache->length = 0;
}