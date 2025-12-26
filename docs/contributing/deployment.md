---
title: Deployment Process and Rollback Procedures
description: Documentation for deploying and maintaining the Physical AI & Humanoid Robotics textbook
---

# Deployment Process and Rollback Procedures

## Overview

This document provides detailed instructions for deploying the Physical AI & Humanoid Robotics textbook website and procedures for rolling back changes if needed.

## Deployment Architecture

### Hosting Platform
- **Platform**: GitHub Pages
- **Domain**: GitHub-provided subdomain or custom domain
- **Build Process**: Automated via GitHub Actions
- **Source**: `main` branch of the repository

### Deployment Flow
1. Changes merged to `main` branch
2. GitHub Actions workflow triggers automatically
3. Site builds using Docusaurus
4. Built site deployed to `gh-pages` branch
5. GitHub Pages serves from `gh-pages` branch

## Prerequisites

### Local Development Environment
- Node.js (v18.x or higher)
- npm or yarn package manager
- Git version control
- GitHub account with repository access

### Required Tools
```bash
# Install Node.js (if not already installed)
# Download from https://nodejs.org/

# Install dependencies
npm install
```

## Deployment Process

### Automated Deployment (Recommended)

The primary deployment method is through GitHub Actions:

1. **Prepare Changes**
   - Create a feature branch
   - Make and test changes locally
   - Commit changes with descriptive messages
   - Push changes to remote repository

2. **Create Pull Request**
   - Submit pull request to `main` branch
   - Ensure all CI checks pass
   - Request code review from team members

3. **Merge to Main**
   - Once approved, merge pull request
   - GitHub Actions will automatically build and deploy

4. **Verify Deployment**
   - Check GitHub Actions workflow status
   - Visit deployed site to verify changes
   - Test navigation and functionality

### Manual Deployment (Emergency)

In case of CI/CD pipeline issues:

```bash
# Clone repository
git clone https://github.com/your-organization/physical-ai-textbook.git
cd physical-ai-textbook

# Install dependencies
npm install

# Build the site
npm run build

# Deploy manually
GIT_USER=<Your GitHub Username> USE_SSH=false npm run deploy
```

## Deployment Configuration

### GitHub Actions Workflow
The deployment workflow is defined in `.github/workflows/ci.yml` and includes:
- Automated testing
- Accessibility checks
- Site building
- Deployment to GitHub Pages

### Environment Variables
- `GIT_USER`: GitHub username for deployment
- `USE_SSH`: Set to false for HTTPS deployment
- `DEPLOYMENT_BRANCH`: Target branch (gh-pages)

## Rollback Procedures

### Automated Rollback

If a deployment causes issues:

1. **Identify the Problem**
   - Monitor site functionality
   - Check GitHub Actions logs
   - Review recent commits

2. **Revert Changes**
   - Go to the GitHub repository
   - Navigate to the `gh-pages` branch
   - Identify the last known good commit
   - Create a new branch from the good commit

3. **Deploy Previous Version**
   ```bash
   # Checkout the good commit
   git checkout <good-commit-hash>

   # Force push to gh-pages (if necessary)
   git push origin +<good-commit-hash>:gh-pages
   ```

### Version-Specific Rollback

#### Rollback via Git Commit
```bash
# Find the commit hash before the problematic changes
git log --oneline -10

# Create a revert commit
git revert <bad-commit-hash>

# Push the revert
git push origin main
```

#### Rollback via Pull Request
1. Create a new branch from a known good state
2. Create a pull request with the revert
3. Review and merge the pull request
4. Monitor the automated deployment

## Monitoring and Verification

### Post-Deployment Checks

After each deployment, verify:

- [ ] Site loads correctly
- [ ] Navigation works properly
- [ ] All links are functional
- [ ] Search functionality works
- [ ] All pages render correctly
- [ ] Mobile responsiveness is maintained
- [ ] Analytics are tracking properly

### Monitoring Tools

#### GitHub Actions
- Check workflow status: `Actions` tab in GitHub repository
- Review logs for any errors or warnings
- Verify successful completion of all steps

#### Site Verification
- Use browser developer tools to check for console errors
- Verify all assets (images, CSS, JS) load correctly
- Test cross-browser compatibility

## Troubleshooting

### Common Deployment Issues

#### Build Failures
- **Symptom**: GitHub Actions workflow fails during build
- **Solution**: Check logs for specific error messages
- **Prevention**: Test builds locally before pushing

#### Content Not Updating
- **Symptom**: New content not appearing on live site
- **Solution**: Verify the deployment workflow completed successfully
- **Check**: Ensure changes were pushed to the correct branch

#### Broken Links
- **Symptom**: 404 errors or broken navigation
- **Solution**: Run `npm run build` locally to identify issues
- **Check**: Verify all internal links use correct paths

### Recovery Procedures

#### Site Completely Down
1. Check GitHub Pages status in repository settings
2. Verify `gh-pages` branch exists and has content
3. Redeploy from a known good commit if necessary

#### Partial Content Missing
1. Check build logs for specific file errors
2. Verify file paths and naming conventions
3. Rebuild and redeploy if needed

## Maintenance Tasks

### Regular Maintenance

#### Weekly
- Review GitHub Actions workflow logs
- Check site functionality
- Update dependencies as needed

#### Monthly
- Review analytics data
- Test mobile responsiveness
- Verify all external links

#### Quarterly
- Update documentation as needed
- Review and update deployment procedures
- Assess performance metrics

### Backup Procedures

#### Content Backup
- All content is stored in the Git repository
- Regular pushes to remote repository provide backup
- Branch protection ensures main branch integrity

#### Configuration Backup
- Docusaurus configuration is version controlled
- Workflow files are stored in repository
- Environment configurations documented in this file

## Security Considerations

### Deployment Security
- Use GitHub Actions secrets for sensitive information
- Limit deployment permissions to authorized personnel
- Regularly review and update access controls

### Content Security
- Validate all user-contributed content
- Sanitize inputs and prevent XSS attacks
- Regular security audits of dependencies

## Performance Optimization

### Build Performance
- Optimize images and assets before deployment
- Use efficient code and minimize bundle sizes
- Implement proper caching strategies

### Site Performance
- Monitor page load times
- Optimize for Core Web Vitals
- Implement lazy loading where appropriate

## Contact Information

For deployment issues:
- **Development Team**: [Development Team Contact]
- **Infrastructure Team**: [Infrastructure Team Contact]
- **Emergency Contact**: [Emergency Contact]

## Appendices

### Appendix A: Command Reference
```bash
# Local development
npm start

# Build site locally
npm run build

# Deploy to GitHub Pages
npm run deploy

# Check content for issues
npx docusaurus check-content
```

### Appendix B: Workflow Status
- **Success**: Site deployed successfully
- **Failure**: Deployment failed, check logs
- **Cancelled**: Deployment cancelled manually
- **Skipped**: Deployment conditions not met

This deployment process ensures consistent, reliable delivery of the Physical AI & Humanoid Robotics textbook to users while maintaining the ability to quickly recover from any issues that may arise.