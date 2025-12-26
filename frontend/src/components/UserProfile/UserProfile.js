import React, { useState, useEffect } from 'react';
import styles from './UserProfile.module.css';

const UserProfile = ({ user, onSave, onCancel }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    fullName: '',
    email: '',
    background: '',
    learningPreferences: {}
  });

  useEffect(() => {
    if (user) {
      setFormData({
        username: user.username || '',
        fullName: user.fullName || '',
        email: user.email || '',
        background: user.background || '',
        learningPreferences: user.learningPreferences || {}
      });
    }
  }, [user]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePreferencesChange = (preference, value) => {
    setFormData(prev => ({
      ...prev,
      learningPreferences: {
        ...prev.learningPreferences,
        [preference]: value
      }
    }));
  };

  const handleSave = () => {
    if (onSave) {
      onSave(formData);
    }
    setIsEditing(false);
  };

  const handleCancel = () => {
    if (user) {
      setFormData({
        username: user.username || '',
        fullName: user.fullName || '',
        email: user.email || '',
        background: user.background || '',
        learningPreferences: user.learningPreferences || {}
      });
    }
    if (onCancel) {
      onCancel();
    }
    setIsEditing(false);
  };

  const toggleEdit = () => {
    setIsEditing(!isEditing);
  };

  const renderViewMode = () => (
    <div className={styles.profileView}>
      <div className={styles.profileHeader}>
        <div className={styles.avatar}>
          <span className={styles.initials}>
            {user?.fullName?.charAt(0)?.toUpperCase() || user?.username?.charAt(0)?.toUpperCase() || 'U'}
          </span>
        </div>
        <div className={styles.userInfo}>
          <h2 className={styles.userName}>{user?.fullName || user?.username || 'User'}</h2>
          <p className={styles.userRole}>{user?.role || 'Student'}</p>
        </div>
        <button className={styles.editButton} onClick={toggleEdit}>
          Edit Profile
        </button>
      </div>

      <div className={styles.profileDetails}>
        <div className={styles.detailGroup}>
          <h3>Account Information</h3>
          <div className={styles.detailItem}>
            <span className={styles.label}>Username:</span>
            <span className={styles.value}>{user?.username}</span>
          </div>
          <div className={styles.detailItem}>
            <span className={styles.label}>Full Name:</span>
            <span className={styles.value}>{user?.fullName || 'Not provided'}</span>
          </div>
          <div className={styles.detailItem}>
            <span className={styles.label}>Email:</span>
            <span className={styles.value}>{user?.email}</span>
          </div>
        </div>

        <div className={styles.detailGroup}>
          <h3>Learning Background</h3>
          <div className={styles.detailItem}>
            <span className={styles.label}>Background:</span>
            <span className={styles.value}>{user?.background || 'Not provided'}</span>
          </div>
        </div>

        <div className={styles.detailGroup}>
          <h3>Learning Preferences</h3>
          {user?.learningPreferences && Object.keys(user.learningPreferences).length > 0 ? (
            Object.entries(user.learningPreferences).map(([key, value]) => (
              <div key={key} className={styles.detailItem}>
                <span className={styles.label}>{key}:</span>
                <span className={styles.value}>{value.toString()}</span>
              </div>
            ))
          ) : (
            <p className={styles.noPreferences}>No learning preferences set</p>
          )}
        </div>

        <div className={styles.detailGroup}>
          <h3>Account Status</h3>
          <div className={styles.statusRow}>
            <span className={styles.statusLabel}>Active:</span>
            <span className={`${styles.statusValue} ${user?.is_active ? styles.active : styles.inactive}`}>
              {user?.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
          <div className={styles.statusRow}>
            <span className={styles.statusLabel}>Verified:</span>
            <span className={`${styles.statusValue} ${user?.is_verified ? styles.verified : styles.unverified}`}>
              {user?.is_verified ? 'Verified' : 'Not Verified'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderEditMode = () => (
    <div className={styles.profileEdit}>
      <div className={styles.profileHeader}>
        <div className={styles.avatar}>
          <span className={styles.initials}>
            {formData.fullName?.charAt(0)?.toUpperCase() || formData.username?.charAt(0)?.toUpperCase() || 'U'}
          </span>
        </div>
        <div className={styles.userInfo}>
          <h2 className={styles.userName}>Edit Profile</h2>
        </div>
      </div>

      <div className={styles.profileForm}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Username</label>
          <input
            type="text"
            value={formData.username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            className={styles.formInput}
            disabled
          />
        </div>

        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Full Name</label>
          <input
            type="text"
            value={formData.fullName}
            onChange={(e) => handleInputChange('fullName', e.target.value)}
            className={styles.formInput}
          />
        </div>

        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Email</label>
          <input
            type="email"
            value={formData.email}
            onChange={(e) => handleInputChange('email', e.target.value)}
            className={styles.formInput}
          />
        </div>

        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Learning Background</label>
          <textarea
            value={formData.background}
            onChange={(e) => handleInputChange('background', e.target.value)}
            className={styles.formTextarea}
            placeholder="Tell us about your technical background and experience..."
            rows={4}
          />
        </div>

        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Learning Preferences</label>
          <div className={styles.preferencesGrid}>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.learning_style === 'visual'}
                  onChange={(e) => handlePreferencesChange('learning_style', e.target.checked ? 'visual' : '')}
                  className={styles.preferenceInput}
                />
                Visual Learner
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.learning_style === 'hands_on'}
                  onChange={(e) => handlePreferencesChange('learning_style', e.target.checked ? 'hands_on' : '')}
                  className={styles.preferenceInput}
                />
                Hands-on Learner
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.learning_pace === 'fast'}
                  onChange={(e) => handlePreferencesChange('learning_pace', e.target.checked ? 'fast' : '')}
                  className={styles.preferenceInput}
                />
                Fast Paced
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.learning_pace === 'moderate'}
                  onChange={(e) => handlePreferencesChange('learning_pace', e.target.checked ? 'moderate' : '')}
                  className={styles.preferenceInput}
                />
                Moderate Paced
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.learning_pace === 'slow'}
                  onChange={(e) => handlePreferencesChange('learning_pace', e.target.checked ? 'slow' : '')}
                  className={styles.preferenceInput}
                />
                Slow Paced
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.focus_area === 'theory'}
                  onChange={(e) => handlePreferencesChange('focus_area', e.target.checked ? 'theory' : '')}
                  className={styles.preferenceInput}
                />
                Theory Focused
              </label>
            </div>
            <div className={styles.preferenceItem}>
              <label className={styles.preferenceLabel}>
                <input
                  type="checkbox"
                  checked={formData.learningPreferences?.focus_area === 'practice'}
                  onChange={(e) => handlePreferencesChange('focus_area', e.target.checked ? 'practice' : '')}
                  className={styles.preferenceInput}
                />
                Practice Focused
              </label>
            </div>
          </div>
        </div>

        <div className={styles.formActions}>
          <button className={styles.cancelButton} onClick={handleCancel}>
            Cancel
          </button>
          <button className={styles.saveButton} onClick={handleSave}>
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className={styles.userProfile}>
      {isEditing ? renderEditMode() : renderViewMode()}
    </div>
  );
};

export default UserProfile;