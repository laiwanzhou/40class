import numpy as np
from aligned_multimodal.stable_routing_structure import RecordingMetadata, build_group_features, build_safe_sessions, fit_source_transition, decode_target

def meta(n, starts, dates=None, users=None):
    return RecordingMetadata(np.array([f"id{i}" for i in range(n)]), np.array(users or ["u"]*n), np.array(dates or ["d"]*n), np.array(starts,float))

def bank(n=4,e=2):
    x=np.zeros((n,e,40));
    for i in range(n): x[i,:,i]=1
    return x

def test_shape_and_base_are_expected():
    f,a=build_group_features(bank(),meta(4,[0,10,100,110]),False); assert f.shape==(4,2*2*40+41); assert not a["users_used"]
def test_permutation_round_trip_is_invariant():
    b=bank(); m=meta(4,[0,10,100,110]); f,_=build_group_features(b,m); p=np.array([2,0,3,1]); mp=RecordingMetadata(m.sample_ids[p],m.users[p],m.dates[p],m.starts[p]); fp,_=build_group_features(b[p],mp); inv=np.argsort(p); assert np.allclose(f,fp[inv])
def test_renaming_ids_users_does_not_change_features():
    b=bank(); f,_=build_group_features(b,meta(4,[0,10,100,110])); m=meta(4,[0,10,100,110],users=["x","y","z","q"]); g,_=build_group_features(b,m); assert np.allclose(f,g)
def test_duplicate_timestamps_have_no_peers():
    f,a=build_group_features(bank(),meta(4,[0,0,100,110]),True); assert a["timestamp_tied_rows"]==2; assert np.all(f[:, -1] == 0)
def test_missing_timestamp_is_retained_and_own_only():
    m=meta(3,[0,np.nan,10]); f,a=build_group_features(bank(3),m,True); assert f.shape[0]==3; assert a["missing_rows_retained"]
def test_user_labels_cannot_change_group_features():
    b=bank(); m=meta(4,[0,10,100,110],users=["a"]*4); g,_=build_group_features(b,m); m2=meta(4,[0,10,100,110],users=["z","y","x","w"]); h,_=build_group_features(b,m2); assert np.allclose(g,h)
def test_date_shift_preserves_structure():
    b=bank(); f,_=build_group_features(b,meta(4,[0,10,100,110],dates=["d"]*4)); g,_=build_group_features(b,meta(4,[0,10,100,110],dates=["new"]*4)); assert np.allclose(f,g)
def test_transition_uses_source_labels_only():
    m=meta(4,[0,10,20,30]); p=np.full((4,40),1/40); p[:,0]=.03; p[:,1]=.03; p[:,2]=.03; p[:,3]=.03
    t=fit_source_transition(np.array([0,1,2,3]),m); u,_=decode_target(p,m,t); t2=fit_source_transition(np.array([9,9,9,9]),m); v,_=decode_target(p,m,t2); assert not np.allclose(t.bigram_log_probability,t2.bigram_log_probability); assert u.shape==v.shape
def test_decode_overlong_session_falls_back_legally():
    n=41; m=meta(n,np.arange(n)); t=fit_source_transition(np.zeros(n,int),m); out,a=decode_target(np.eye(40)[np.arange(n)%40],m,t); assert len(out)==n; assert a["fallback_sessions_over_40"]>=1
