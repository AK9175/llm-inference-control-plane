package auth

import "testing"

func TestKeyStore_AddAndLookup(t *testing.T) {
	s := NewKeyStore()
	s.AddKey("secret-1", KeyInfo{KeyID: "customer-a", RequestsPerMin: 60})

	info, ok := s.Lookup("secret-1")
	if !ok {
		t.Fatal("expected key to be found")
	}
	if info.KeyID != "customer-a" {
		t.Errorf("KeyID: got %s, want customer-a", info.KeyID)
	}
	if info.RequestsPerMin != 60 {
		t.Errorf("RequestsPerMin: got %d, want 60", info.RequestsPerMin)
	}
}

func TestKeyStore_LookupMissing_ReturnsFalse(t *testing.T) {
	s := NewKeyStore()
	_, ok := s.Lookup("never-registered")
	if ok {
		t.Error("expected ok=false for an unregistered key")
	}
}

func TestKeyStore_RemoveKey(t *testing.T) {
	s := NewKeyStore()
	s.AddKey("secret-1", KeyInfo{KeyID: "customer-a"})
	s.RemoveKey("secret-1")

	_, ok := s.Lookup("secret-1")
	if ok {
		t.Error("expected key to be gone after RemoveKey")
	}
}

func TestKeyStore_AddKey_OverwritesExisting(t *testing.T) {
	s := NewKeyStore()
	s.AddKey("secret-1", KeyInfo{KeyID: "old", RequestsPerMin: 10})
	s.AddKey("secret-1", KeyInfo{KeyID: "new", RequestsPerMin: 99})

	info, _ := s.Lookup("secret-1")
	if info.KeyID != "new" || info.RequestsPerMin != 99 {
		t.Errorf("got %+v, want KeyID=new RequestsPerMin=99", info)
	}
}
