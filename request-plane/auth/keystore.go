// Package auth authenticates clients via API key and enforces a
// per-key rate budget — the gateway's outermost gate, checked before any
// other work (queueing, SLO estimation, dispatch) happens.
package auth

import "sync"

// KeyInfo describes one registered API key's permissions.
type KeyInfo struct {
	KeyID string
	// RequestsPerMin is the rate limit budget. 0 means unlimited.
	RequestsPerMin int
}

// KeyStore validates API keys and returns their metadata. Safe for
// concurrent lookups from request-handling goroutines alongside occasional
// provisioning/revocation writes.
type KeyStore struct {
	mu   sync.RWMutex
	keys map[string]KeyInfo
}

// NewKeyStore creates an empty KeyStore.
func NewKeyStore() *KeyStore {
	return &KeyStore{keys: make(map[string]KeyInfo)}
}

// AddKey registers or replaces an API key's info.
func (s *KeyStore) AddKey(apiKey string, info KeyInfo) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keys[apiKey] = info
}

// RemoveKey revokes an API key — subsequent Lookups fail.
func (s *KeyStore) RemoveKey(apiKey string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.keys, apiKey)
}

// Lookup returns the key's info and whether it is currently registered.
func (s *KeyStore) Lookup(apiKey string) (KeyInfo, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	info, ok := s.keys[apiKey]
	return info, ok
}
