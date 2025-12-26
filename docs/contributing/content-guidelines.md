---
title: Content Authoring Guidelines
description: Guidelines for contributing content to the Physical AI & Humanoid Robotics textbook
---

# Content Authoring Guidelines

## Overview

This document provides guidelines for contributing content to the Physical AI & Humanoid Robotics textbook. These guidelines ensure consistency, quality, and maintainability of the educational content.

## Content Structure

### Frontmatter Requirements

All content pages must include the following frontmatter:

```yaml
---
title: Descriptive Title
description: Brief description of the content
module: [1-4]  # Module number
chapter: [1-10]  # Chapter number if applicable
learning_objectives:  # Required for chapters
  - Objective 1
  - Objective 2
  - Objective 3
difficulty: [beginner|intermediate|advanced]
estimated_time: [minutes]  # Time to complete the chapter
tags:  # Relevant tags
  - tag1
  - tag2
authors:  # Author information
  - Textbook Team
prerequisites:  # Prerequisites for the content
  - List of prerequisites
---
```

### Content Organization

1. **Introduction**: Brief overview of the topic and its relevance
2. **Learning Objectives**: Clear, measurable learning outcomes
3. **Main Content**: Well-structured with appropriate headings
4. **Examples**: Practical examples with code snippets where applicable
5. **Summary**: Key takeaways and concepts
6. **Learning Check**: Self-assessment questions

## Writing Style

### Tone and Voice

- Use an educational, approachable tone
- Write in active voice when possible
- Address the reader directly ("you" rather than "the student")
- Be concise but thorough

### Technical Content

- Define technical terms when first used
- Use consistent terminology throughout
- Include relevant code examples with explanations
- Provide context for code snippets

### Headings and Structure

- Use proper heading hierarchy (H1 for main title, H2 for sections, etc.)
- Limit sections to manageable chunks
- Use descriptive heading titles

## Code Examples

### Formatting

```python
# Python examples
def example_function(param1, param2):
    """
    Brief description of the function.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Returns:
        Description of return value
    """
    # Implementation here
    return result
```

```cpp
// C++ examples
class ExampleClass {
public:
    // Constructor
    ExampleClass(int value) : member_value(value) {}

    // Method implementation
    void doSomething() {
        // Implementation here
    }

private:
    int member_value;
};
```

### Best Practices

- Include comments explaining complex logic
- Use meaningful variable and function names
- Follow language-specific style guides
- Include error handling where appropriate

## Markdown Guidelines

### Text Formatting

- Use **bold** for emphasis on important terms
- Use *italics* for book titles or to emphasize concepts
- Use `inline code` for variable names, function names, and commands
- Use ```code blocks``` for multi-line code examples

### Lists

Use ordered lists when sequence matters, unordered lists for general items:

```markdown
1. First step in the process
2. Second step in the process
3. Final step in the process

- General item 1
- General item 2
- General item 3
```

### Links and References

- Use descriptive link text instead of raw URLs
- Link to internal content using relative paths
- Use external links sparingly and only to authoritative sources

## Content Quality Standards

### Accuracy

- Verify all technical information
- Test code examples when possible
- Cite sources for claims and data
- Ensure examples work as described

### Completeness

- Cover all learning objectives
- Provide sufficient detail for understanding
- Include practical applications
- Address common questions or misconceptions

### Consistency

- Use consistent terminology throughout
- Follow the same structural patterns
- Maintain consistent formatting
- Use parallel construction where appropriate

## Media Guidelines

### Images

- Use descriptive alt text for accessibility
- Include captions when necessary
- Ensure images are high resolution
- Compress images to optimize loading time

### Diagrams

- Use consistent styling
- Include legends when necessary
- Ensure diagrams are readable at different sizes
- Prefer SVG format when possible

## Review Process

### Self-Review Checklist

Before submitting content, ensure:

- [ ] All frontmatter fields are completed
- [ ] Learning objectives are clear and measurable
- [ ] Content aligns with learning objectives
- [ ] Code examples are correct and well-commented
- [ ] Grammar and spelling are correct
- [ ] All links are functional
- [ ] Images have appropriate alt text
- [ ] Content is accessible and inclusive

### Peer Review

All content should undergo peer review focusing on:

- Technical accuracy
- Clarity of explanations
- Effectiveness of examples
- Alignment with learning objectives
- Accessibility considerations

## Contribution Workflow

### Creating New Content

1. Fork the repository
2. Create a new branch for your content
3. Follow the content structure guidelines
4. Test code examples
5. Submit a pull request with a clear description

### Updating Existing Content

1. Identify the specific improvements needed
2. Make changes following the guidelines
3. Update any affected cross-references
4. Test that changes don't break existing functionality

## Technical Requirements

### File Naming

- Use descriptive, lowercase filenames with hyphens
- Follow the pattern: `topic-name.md`
- For chapters: `chapter-{module}/{chapter-number}-{topic}.md`

### File Structure

- Place chapter content in `docs/chapter-{module}/`
- Place instructor resources in `docs/instructor/`
- Place contributing docs in `docs/contributing/`

## Accessibility Guidelines

### Inclusive Language

- Use gender-neutral language
- Avoid idioms that may not translate well
- Consider diverse backgrounds and experiences

### Content Accessibility

- Provide alternative text for images
- Use sufficient color contrast
- Include captions for multimedia
- Structure content with proper headings

## Review and Update Schedule

Content should be reviewed and updated:

- Annually for major updates
- Biannually for minor corrections
- As needed when technology changes
- After user feedback indicates issues

## Getting Help

For questions about content authoring:

- Review existing content for examples
- Check the FAQ section
- Contact the editorial team
- Participate in authoring workshops

These guidelines help ensure that all content in the Physical AI & Humanoid Robotics textbook maintains high quality and provides an excellent learning experience for students.